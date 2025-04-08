FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY fft_example.py .

# Create output directory for generated files
RUN mkdir -p output

# Run the FFT example script
ENTRYPOINT ["python", "fft_example.py"]