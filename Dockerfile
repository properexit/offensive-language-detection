FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py", "--lang", "english", "--task", "A", "--config", "config/base.yaml"]