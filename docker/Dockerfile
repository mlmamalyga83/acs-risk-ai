FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

EXPOSE 8501

CMD ["streamlit", "run", "src/app/main.py", "--server.port=8501"]
