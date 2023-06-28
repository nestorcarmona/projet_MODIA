# Build the gradio image
docker build -t gradio:latest -f gradio.Dockerfile .
# Build the predict image
docker build -t main:latest -f main.Dockerfile .