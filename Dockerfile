FROM python:3.10-slim

WORKDIR /app

# Install basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Upgrade pip and install requirements
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose the default Streamlit port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
