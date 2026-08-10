FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace/Oncer/api

# Install any missing system libraries for imgaug/OpenCV
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Works for both LF and CRLF line endings
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Run backend once container's built
ENTRYPOINT ["./entrypoint.sh"]
