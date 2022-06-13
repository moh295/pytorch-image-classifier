FROM simple-classifier-base
WORKDIR /App
COPY . /App
ENTRYPOINT ["python3","app300x2.py"]