FROM simple-classifier-base
WORKDIR /App
COPY . /App
#ENTRYPOINT ["python3","app_fasterrcnn_mobilenet_torchvision.py"]
ENTRYPOINT ["python3","app_pennFudan.py"]