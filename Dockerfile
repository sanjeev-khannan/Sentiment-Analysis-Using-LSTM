# Use a base image with the necessary dependencies
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .
COPY src /app/src/
COPY checkpoints /app/checkpoints/

# Expose the port your application will run on
EXPOSE 8082

# Define the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
