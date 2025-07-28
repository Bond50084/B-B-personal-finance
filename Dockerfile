# Use an official Python runtime as a parent image.
# We'll use Python 3.9 as it's a stable version.
# You can try 3.10 or 3.11 if you prefer, but ensure compatibility with your libraries.
FROM python:3.9-slim-buster

# Set the working directory in the container to /app.
# All subsequent commands will be run from this directory inside the container.
WORKDIR /app

# Install Python dependencies.
# We copy requirements.txt first to leverage Docker's build cache.
# If requirements.txt doesn't change, Docker won't re-run pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container.
# The '.' here refers to the current directory where the Dockerfile is located
# (which is the root of your 'your-financial-sim/' project).
# This will copy app.py, simulation_v2_inflation.py, templates/, and static/.
COPY . .

# Cloud Run requires that the application listen for requests on the port defined by the PORT environment variable.
# The value of PORT is always set to 8080 by Cloud Run.
# We set this environment variable here.
ENV PORT 8080
EXPOSE 8080 

# Command to run the Flask application with Gunicorn.
# Gunicorn is a production-ready WSGI HTTP server for Python web applications.
# It's recommended over Flask's built-in development server for production.
# 'app:app' refers to the 'app' Flask instance within the 'app.py' file.
# '--bind 0.0.0.0:8080' tells Gunicorn to listen on all network interfaces on port 8080.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]