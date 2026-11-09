FROM python:3.12-slim
WORKDIR /app
COPY dist/Continental_OTA_Dashboard.html /app/index.html
ENV PYTHONUNBUFFERED=1
CMD python -m http.server ${PORT:-8080} --bind 0.0.0.0 --directory /app
