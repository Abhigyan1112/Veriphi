# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent interactive prompts during installation
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for dlib and opencv-python
# This step is crucial as these libraries have non-Python dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
# This assumes your Dockerfile is in the root of your project.
COPY . .

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Define the command to run your app using Gunicorn, a production-grade WSGI server.
# This is the standard way to run Flask apps in production.
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]
