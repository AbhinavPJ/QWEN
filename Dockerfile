# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Set cache directory to a universally writable temp location
ENV HF_HOME /tmp/cache
ENV TRANSFORMERS_CACHE /tmp/cache
ENV HF_HUB_CACHE /tmp/cache

# Update package lists and install git
RUN apt-get update && apt-get install -y git

# === THIS LINE IS NOW FIXED ===
# Install torch, torchvision, AND transformers from GitHub
RUN pip install --no-cache-dir torch torchvision git+https://github.com/huggingface/transformers
# ===============================

# Copy and install the rest of your requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Tell Docker what port the app will run on
EXPOSE 7860

# Command to run your app
CMD ["python", "app.py"]