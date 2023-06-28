FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install the requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy the main.py script
COPY ./main.py /app/main.py
# Copy the NFC model script
COPY /NCF.py /app/NCF.py
# Copy the NCF model weights
COPY ./checkpoints/weights.pth /app/weights.pth
# Copy the test data
COPY ./data/test_script.csv /app/test_script.csv

WORKDIR /app

EXPOSE 7860

CMD ["python", "main.py", "--model_path", "/app/weights.pth", "--test_script_path", "/app/test_script.csv"]

