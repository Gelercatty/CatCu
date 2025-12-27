docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  nvidia/cuda:13.0.1-devel-ubuntu22.04 \
  bash