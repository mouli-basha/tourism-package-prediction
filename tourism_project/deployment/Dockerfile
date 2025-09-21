# tourism_project/deployment/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py ./
# HF Spaces default port for Streamlit
EXPOSE 7860

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
