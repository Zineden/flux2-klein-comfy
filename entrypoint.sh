#!/bin/bash
set -e

echo "Checking CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK', torch.__version__)"
export CUDA_VISIBLE_DEVICES=0

# ComfyUI 백그라운드 기동 (FLUX.2 GGUF 편집용)
echo "Starting ComfyUI..."
python /ComfyUI/main.py --listen 127.0.0.1 --port 8188 &

echo "Waiting for ComfyUI..."
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then echo "ComfyUI ready"; break; fi
  sleep 2
done

echo "Starting handler..."
exec python -u /handler.py
