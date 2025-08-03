FROM python:3.10-slim

WORKDIR /app

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Set environment variable to avoid usage stats error
ENV STREAMLIT_HOME=/app

EXPOSE 7860

# Set up .streamlit config in a writable directory and run
CMD mkdir -p /app/.streamlit && \
    echo "\
[general]\n\
email = \"\"\n\
" > /app/.streamlit/config.toml && \
    streamlit run app.py --server.port=7860 --server.address=0.0.0.0
