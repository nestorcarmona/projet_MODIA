FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy the recommender app script
COPY ./recommender_app.py /app/recommender_app.py
# Copy the binary classifier script
COPY ./BinaryClassifier.py /app/BinaryClassifier.py
# Copy the word2vec model
COPY ./checkpoints/word2vec.pkl /app/word2vec.pkl
# Copy the binary classifier model
COPY ./checkpoints/binary_classifier.pth /app/binary_classifier.pth

EXPOSE 7860

WORKDIR /app

CMD ["python", "recommender_app.py", "--word2vec_path", "/app/word2vec.pkl", "--model_path", "/app/binary_classifier.pth"]
