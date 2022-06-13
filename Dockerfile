FROM simple-classifier-base
WORKDIR /App
COPY . /App
ENTRYPOINT ["python3","app32x2.py"]