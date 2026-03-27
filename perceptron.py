"""
Multi-layer Perceptron trained on MNIST using Keras.
Training metrics are logged to TensorBoard.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_data():
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten 28x28 images to 784-dimensional vectors and normalise to [0, 1]
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_model(input_dim: int = 784, num_classes: int = 10) -> keras.Model:
    """Build a multi-layer perceptron model."""
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(input_dim,), name="hidden_1"),
            layers.Dense(64, activation="relu", name="hidden_2"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="perceptron",
    )
    return model


def main():
    # Directories
    log_dir = os.environ.get("LOG_DIR", "./logs")
    model_dir = os.environ.get("MODEL_DIR", "./model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Hyper-parameters (can be overridden via environment variables)
    epochs = int(os.environ.get("EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))

    print("TensorFlow version:", tf.__version__)
    print(f"Epochs: {epochs}  |  Batch size: {batch_size}")
    print(f"TensorBoard log directory: {log_dir}")

    # Data
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Training samples : {len(x_train)}")
    print(f"Test samples     : {len(x_test)}")

    # Model
    model = build_model()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
    )

    # Training
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tensorboard_cb],
    )

    # Evaluation
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss    : {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Save the trained model
    model_path = os.path.join(model_dir, "perceptron.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
