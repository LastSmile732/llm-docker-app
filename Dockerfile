# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Set the working directory in the container to /app
WORKDIR /app

# COPY /sbox/etc/server.crt /app/server.crt
# COPY /sbox/etc/server.key /app/server.key

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install required packages from requirements.txt
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the ports for FastAPI and Streamlit
EXPOSE 9999
EXPOSE 8501

# Copy and give execute permissions to the start script
COPY start_server.sh /app/
RUN chmod +x /app/start_server.sh

# Run the start script
CMD ["/app/start_server.sh"]
