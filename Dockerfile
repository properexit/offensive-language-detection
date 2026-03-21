FROM python:3.10-slim

WORKDIR /app

# install dependencies 
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# default run 
CMD ["python", "main.py", "--lang", "english", "--task", "A", "--config", "config/base.yaml"]