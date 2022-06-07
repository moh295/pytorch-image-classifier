FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
WORKDIR /App
COPY . /App


RUN python3 -m pip install opencv-python
ENTRYPOINT ["python3","app.py"]