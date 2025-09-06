# Use an official slim Python image as a base
FROM python:3.9-slim

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by pyzbar
# This command is crucial to fix the "Unable to find zbar shared library" error
RUN apt-get update && apt-get install -y libzbar0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port your application will run on
# The logs show gunicorn listening on port 3000
EXPOSE 3000

# Define the command to run your application
# Replace 'scanner.py' with your actual file name if it's different.
# The command tells gunicorn to run the 'app' object from the 'scanner' file.
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "scanner:app"]