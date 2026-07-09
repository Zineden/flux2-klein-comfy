# flux2-klein-comfy

RunPod serverless worker for **FLUX.2-klein-4B image editing** via **ComfyUI + GGUF**.
Input image + prompt → edited image (e.g. "remove all texts from the image").

**Why this build**
- **FLUX.2-klein-4B is Apache-2.0** (free commercial use) — unlike FLUX.1-Kontext-dev (non-commercial).
- **ComfyUI + GGUF (Q8) + fp4 text encoder** → ~7 GB of weights (vs ~13 GB bf16 diffusers) → faster cold-host transfer.
- Built on **`wlsdml1114/multitalk-base:1.7`** (torch 2.7 / cuda 12.8) — the fleet-cached base used by the qwen and FLUX.1-Kontext hub workers, so base layers are already on most RunPod hosts.
- **Distilled klein-4B → 4 steps, cfg 1** (few-step, fast inference).

## API

`event["input"]`:

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | ✅ | — | edit instruction |
| `image_url` / `image_base64` / `image` | string | ✅ | — | input image (URL fetched with a browser UA so alicdn etc. don't 420) |
| `num_inference_steps` | int | — | 4 | klein distilled |
| `cfg` | float | — | 1.0 | klein distilled |
| `seed` | int | — | random | |

**Output:** `{ "image_url": "https://<r2>/klein/<sha256>.png" }` when R2 env is set (the worker uploads to
Cloudflare R2), else `{ "image": "data:image/png;base64,..." }`. On error `{ "error": "..." }`.

**R2 env** (same as the bg/qwen workers): `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_PUBLIC_BASE`, optional `R2_PREFIX` (default `klein`).

## Models (baked)
- diffusion: `unsloth/FLUX.2-klein-4B-GGUF` → `flux-2-klein-4b-Q8_0.gguf` (`models/unet/`)
- text encoder: `Comfy-Org/vae-text-encorder-for-flux-klein-4b` → `qwen_3_4b_fp4_flux2.safetensors` (`models/text_encoders/`)
- vae: `flux2-vae.safetensors` (`models/vae/`)

## Publish to RunPod Hub (fleet-cached image → ~24 s cold start)
`.runpod/hub.json` + `.runpod/tests.json` are included. Push to GitHub, then in the RunPod console →
**Hub → publish this repo**. RunPod builds it, runs the tests, and pre-distributes the image across its
host fleet — that fleet caching is what gives hub workers their fast cold starts.

## Notes
- Editing uses FLUX.2's `ReferenceLatent` chain (not img2img); output size follows the input image.
- Needs a recent ComfyUI (FLUX.2 nodes) — pulled from `master` at build. `ComfyUI-GGUF` (city96) provides `UnetLoaderGGUF`.
