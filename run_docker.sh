docker run --gpus all -it \
  --name cudadev \
  --ipc=host \
  -v /home/mayanwen/CatCu:/workspace \
  -w /workspace \
  cudadev:latest