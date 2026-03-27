# perceptron-keras

A multi-layer perceptron (MLP) implemented with **Keras / TensorFlow**, fully containerised with **Docker**, and instrumented with **TensorBoard** for live training visualisation.

---

## Project structure

```
.
├── perceptron.py       # Keras MLP model + training loop
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
└── docker-compose.yml  # Train + TensorBoard services
```

---

## Quick start with Docker Compose

### 1. Build the image and train the model

```bash
docker compose up train
```

This will:
- Download the MNIST dataset (cached inside the container).
- Train a two-hidden-layer MLP for 10 epochs.
- Write TensorBoard logs to `./logs/`.
- Save the trained model to `./model/perceptron.keras`.

### 2. Launch TensorBoard

```bash
docker compose up tensorboard
```

Then open <http://localhost:6006> in your browser to explore loss and accuracy curves, the computation graph, and weight histograms.

---

## Running both services together

```bash
docker compose up
```

---

## Configuration

Override any of the following environment variables (in `docker-compose.yml` or via `-e`):

| Variable     | Default | Description                        |
|--------------|---------|------------------------------------|
| `EPOCHS`     | `10`    | Number of training epochs          |
| `BATCH_SIZE` | `128`   | Mini-batch size                    |
| `LOG_DIR`    | `./logs`  | TensorBoard log directory        |
| `MODEL_DIR`  | `./model` | Where the saved model is written |

Example — train for 20 epochs:

```bash
docker compose run -e EPOCHS=20 train
```

---

## Running locally (without Docker)

```bash
pip install -r requirements.txt
python perceptron.py
tensorboard --logdir=logs
```

---

## Model architecture

```
Input (784)  →  Dense(128, ReLU)  →  Dense(64, ReLU)  →  Dense(10, Softmax)
```

Trained with the **Adam** optimizer and **sparse categorical cross-entropy** loss on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (60 000 training / 10 000 test images).