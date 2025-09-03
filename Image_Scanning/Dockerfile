# Use an official Python runtime as a parent image. python:3.9-slim is a good balance of size and compatibility.
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering stdout/stderr and to disable interactive prompts during package installation.
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies. This is the most critical step for dlib and opencv-python.
# - build-essential & cmake: Needed to compile dlib from source.
# - libgl1-mesa-glx, libsm6, libxext6: Required by OpenCV for image processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container. All subsequent commands will run from here.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt.
# --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code (app.py, static/, templates/) into the container.
COPY . .

# Expose port 3000 to allow traffic to the container.
EXPOSE 3000

# Define the command to run your application using Gunicorn, a production-grade WSGI server.
# This is the standard for running Flask apps in production.
# It binds the server to 0.0.0.0:3000, making it accessible from outside the container.
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]

