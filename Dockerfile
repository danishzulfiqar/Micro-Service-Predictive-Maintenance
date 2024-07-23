# Use the official Python 3.10.14 image from the Docker Hub
FROM python:3.10.14-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app
COPY ./Models /code/Models

# 
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
