FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace/Oncer/api

# Install any missing system libraries for imgaug/OpenCV
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install ALL Python dependencies listed in pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir -r ".[training]"

# Copy backend code
COPY . .

# Make entrypoint.sh executable and handle both line endings (LF, CRLF)
RUN chmod +x entrypoint.sh && sed -i 's/\r$//' entrypoint.sh  

# Run backend once container's built
ENTRYPOINT ["./entrypoint.sh"]