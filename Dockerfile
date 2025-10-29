FROM python:3.10-slim
WORKDIR /code
ENV HF_HOME /tmp/cache
ENV TRANSFORMERS_CACHE /tmp/cache
ENV HF_HUB_CACHE /tmp/cache
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]