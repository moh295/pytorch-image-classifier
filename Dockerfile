FROM simple-classifier-base
WORKDIR /App
COPY . /App
ENTRYPOINT ["python3","fasterrcnn_torchvision.py"]