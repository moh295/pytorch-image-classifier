FROM simple-classifier-base
WORKDIR /App
COPY . /App
ENTRYPOINT ["python3","app128x2.py"]