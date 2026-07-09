"""RunPod ComfyUI serverless worker — FLUX.2-klein-4B (GGUF) image editing.

- 입력 이미지(alicdn URL / base64) + 프롬프트 → 편집 이미지(예: 텍스트 제거).
- ComfyUI 워크플로(workflow.json)를 웹소켓으로 실행. 모델은 이미지에 구운 GGUF/fp4(작음 → 콜드스타트↓).
- 결과는 Cloudflare R2에 업로드하고 image_url을 반환(서버 klein-status가 그대로 사용). R2 미설정 시 base64.

Input (event["input"]):
  prompt              (str, required)  편집 지시(예: "remove all texts from the image")
  image_url | image_base64 | image  (str)  입력 이미지(URL 또는 base64/경로)
  num_inference_steps (int, opt=4)   distilled klein은 4스텝
  cfg                 (float, opt=1)  distilled klein은 cfg=1
  seed                (int, opt)      미지정 시 랜덤
Output:
  { "image_url": "https://<r2>/edits/....png" }  또는  { "image": "data:image/png;base64,..." }  또는 { "error": ... }
"""

import os
import io
import time
import json
import uuid
import base64
import hashlib
import logging
import threading
import urllib.parse
import urllib.request
import urllib.error

import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())
COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", "/ComfyUI/input")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/workflow.json")

# --- R2 (bg/qwen 워커와 동일 env 이름) ---
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE")
R2_PREFIX = os.environ.get("R2_PREFIX", "klein")
_s3 = None
_s3_lock = threading.Lock()

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def r2_enabled():
    return all([R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY, R2_PUBLIC_BASE])


def get_s3():
    global _s3
    if _s3 is None:
        with _s3_lock:
            if _s3 is None:
                import boto3
                _s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT,
                                   aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY)
    return _s3


def upload_r2(png_bytes):
    key = f"{R2_PREFIX.strip('/')}/{hashlib.sha256(png_bytes).hexdigest()}.png"
    get_s3().put_object(Bucket=R2_BUCKET, Key=key, Body=png_bytes, ContentType="image/png",
                        CacheControl="public, max-age=31536000, immutable")
    return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"


def fetch_bytes(url):
    # alicdn 등 핫링크/레이트 차단(420) 회피 — 브라우저 UA 필수.
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Referer": "https://detail.1688.com/",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    })
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e; time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"image download failed ({url[:120]}): {last}")


def resolve_input_image(job_input):
    """image_url(다운로드) / image_base64 / image(경로|base64) → ComfyUI input 디렉토리에 저장한 파일명 반환."""
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    name = f"in_{uuid.uuid4().hex}.png"
    dst = os.path.join(COMFY_INPUT_DIR, name)
    url = job_input.get("image_url")
    b64 = job_input.get("image_base64")
    generic = job_input.get("image")
    if url:
        data = fetch_bytes(str(url))
    elif b64:
        data = base64.b64decode(str(b64).split(",", 1)[-1])
    elif isinstance(generic, str) and generic:
        if generic.startswith("http://") or generic.startswith("https://"):
            data = fetch_bytes(generic)
        else:
            try:
                data = base64.b64decode(generic.split(",", 1)[-1])
            except Exception:
                return generic  # 이미 존재하는 파일명/경로로 간주
    else:
        raise ValueError("이미지(image_url/image_base64/image)가 필요합니다.")
    with open(dst, "wb") as f:
        f.write(data)
    return name


# ── ComfyUI HTTP/WS helpers (kontext 워커와 동일 패턴) ──
def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    req = urllib.request.Request(f"http://{server_address}:8188/prompt", data=json.dumps(p).encode("utf-8"))
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        # ComfyUI는 워크플로 검증 실패 시 400 + 상세 JSON(어느 노드/입력이 문제인지)을 준다 → 그대로 노출.
        body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"ComfyUI /prompt {e.code}: {body[:1200]}")


def get_image(filename, subfolder, folder_type):
    q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    with urllib.request.urlopen(f"http://{server_address}:8188/view?{q}") as r:
        return r.read()


def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}:8188/history/{prompt_id}") as r:
        return json.loads(r.read())


def run_workflow(ws, prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    while True:
        out = ws.recv()
        if isinstance(out, str):
            m = json.loads(out)
            if m.get("type") == "executing":
                d = m["data"]
                if d.get("node") is None and d.get("prompt_id") == prompt_id:
                    break
    hist = get_history(prompt_id)[prompt_id]
    for node_id, node_output in hist.get("outputs", {}).items():
        for image in node_output.get("images", []):
            return get_image(image["filename"], image["subfolder"], image["type"])
    return None


def wait_for_comfy():
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://{server_address}:8188/", timeout=5)
            return True
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI 서버에 연결할 수 없습니다.")


def handler(job):
    try:
        import websocket
        job_input = job.get("input", {})
        prompt_text = job_input.get("prompt")
        if not prompt_text:
            return {"error": "prompt is required"}

        image_name = resolve_input_image(job_input)
        steps = int(job_input.get("num_inference_steps", job_input.get("steps", 4)))
        cfg = float(job_input.get("cfg", 1.0))   # distilled klein → cfg 1
        seed = job_input.get("seed")

        wf = json.load(open(WORKFLOW_PATH, "r"))
        wf["20"]["inputs"]["image"] = image_name
        wf["30"]["inputs"]["text"] = str(prompt_text)
        wf["41"]["inputs"]["steps"] = steps
        wf["44"]["inputs"]["cfg"] = cfg
        if seed is not None:
            wf["43"]["inputs"]["noise_seed"] = int(seed)
            wf["43"]["inputs"]["control_after_generate"] = "fixed"

        wait_for_comfy()
        ws = websocket.WebSocket()
        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        for attempt in range(36):
            try:
                ws.connect(ws_url); break
            except Exception as e:
                if attempt == 35:
                    raise RuntimeError(f"웹소켓 연결 실패: {e}")
                time.sleep(5)
        png = run_workflow(ws, wf)
        ws.close()

        if not png:
            return {"error": "이미지를 생성하지 못했습니다."}
        if r2_enabled():
            try:
                return {"image_url": upload_r2(png)}
            except Exception as e:
                logger.error(f"R2 업로드 실패, base64 폴백: {e}")
        return {"image": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}
    except Exception as e:
        logger.exception("handler error")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
