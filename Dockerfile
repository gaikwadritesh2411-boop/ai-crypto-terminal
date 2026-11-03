FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh && mkdir -p data
# On a host with a persistent disk, mount it at /app/data so the paper account survives restarts.
EXPOSE 8501
CMD ["./start.sh"]
