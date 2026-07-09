# FLUX.2-klein-4B (GGUF) ComfyUI serverless worker — 이미지 편집(텍스트 제거 등).
# 베이스: wlsdml1114/multitalk-base:1.7 (torch 2.7.0+cu128, cuda12.8) = qwen/flux-kontext 워커와 동일.
#   → RunPod 호스트 fleet에 레이어가 캐시되어 있어 콜드스타트가 빠르다.
# 모델은 GGUF(Q8)/fp4로 구워 이미지가 작다(~7GB) → 콜드호스트 전송 시간 단축.
# klein-4B는 Apache-2.0(상업 이용 가능).
FROM wlsdml1114/multitalk-base:1.7 as runtime

RUN apt-get update && apt-get install -y --no-install-recommends git wget curl && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install -U "huggingface_hub[hf_transfer]" && \
    pip install --no-cache-dir runpod websocket-client boto3

WORKDIR /

# ComfyUI 최신(FLUX.2 노드: flux2 CLIP type, EmptyFlux2LatentImage, Flux2Scheduler) + ComfyUI-GGUF
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    pip install --no-cache-dir --upgrade gguf

RUN mkdir -p /ComfyUI/models/unet /ComfyUI/models/text_encoders /ComfyUI/models/vae /ComfyUI/input

# ── 모델 굽기(작은 양자화 파일들) ──
# diffusion(GGUF Q8, distilled) → models/unet/  (파일이 레포 최상위라 평평하게 저장됨)
RUN hf download unsloth/FLUX.2-klein-4B-GGUF flux-2-klein-4b-Q8_0.gguf --local-dir /ComfyUI/models/unet/
# text encoder(Qwen3-4B, fp4) + vae → split_files/ 하위라 받은 뒤 평평한 경로로 이동
RUN hf download Comfy-Org/vae-text-encorder-for-flux-klein-4b split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors --local-dir /tmp/klein_dl && \
    mv /tmp/klein_dl/split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors /ComfyUI/models/text_encoders/ && \
    hf download Comfy-Org/vae-text-encorder-for-flux-klein-4b split_files/vae/flux2-vae.safetensors --local-dir /tmp/klein_dl && \
    mv /tmp/klein_dl/split_files/vae/flux2-vae.safetensors /ComfyUI/models/vae/ && \
    rm -rf /tmp/klein_dl

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
