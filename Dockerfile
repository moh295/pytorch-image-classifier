FROM simple-classifier-base
WORKDIR /App
RUN pip3 install cython
RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

COPY . /App
#ENTRYPOINT ["python3","app_fasterrcnn_mobilenet_torchvision.py"]
ENTRYPOINT ["python3","app_pennFudan.py"]