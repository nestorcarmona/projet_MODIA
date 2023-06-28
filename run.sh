# Run the gradio image (with GPU support + port forwarding)
docker run --gpus=all --ipc=host -p 7860:7860 gradio:latest
# Run the predict image (with GPU support)
docker run --gpus=all --ipc=host main:latest