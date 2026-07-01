FROM python:3.12-slim

# Install system dependencies needed for LightGBM, git, and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and assets
COPY src/ ./src/
COPY app/ ./app/
COPY notebooks/ ./notebooks/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY results/ ./results/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify container is active
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit dashboard app
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
