# Use the official TensorFlow image as the base
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY perceptron.py .

# Expose TensorBoard port
EXPOSE 6006

# Default command: train the model
CMD ["python", "perceptron.py"]
