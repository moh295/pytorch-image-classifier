FROM simple-classifier-base
WORKDIR /App
COPY . /App

run export PYTHONPATH="${PYTHONPATH}:/App/Mymodel"
ENTRYPOINT ["python3","app.py"]