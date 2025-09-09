FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY run.py .
COPY flask_template.html .

# Create outputs directory
RUN mkdir -p outputs

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]