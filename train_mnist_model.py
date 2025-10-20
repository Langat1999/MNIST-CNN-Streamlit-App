# ===========================================
# 🧠 MNIST CNN MODEL TRAINING SCRIPT
# ===========================================
# Trains a Convolutional Neural Network (CNN) on MNIST digits (0–9)
# Saves the trained model as mnist_cnn_model.h5
# Compatible with Streamlit MNIST app
# ===========================================

import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import argparse

# -----------------------------
# 1️⃣ Load and Preprocess Data
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape to (num_samples, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# 1.1️⃣ Reproducibility & CLI Args
# -----------------------------
parser = argparse.ArgumentParser(description="Train MNIST CNN model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate")
parser.add_argument("--val_split", type=float, default=0.1667, help="Validation fraction from train (~10k/60k)")
parser.add_argument("--model_dir", type=str, default="artifacts/mnist_saved_model", help="Directory to save SavedModel")
parser.add_argument("--model_h5", type=str, default="mnist_cnn_model.h5", help="Filename for Keras H5 model")
args, _ = parser.parse_known_args()

seed_value = 42
os.environ["PYTHONHASHSEED"] = str(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Enable mixed precision on supported GPUs for speedups
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

# -----------------------------
# 1.2️⃣ Train/Val/Test Splits & tf.data Pipelines
# -----------------------------
num_train = x_train.shape[0]
val_size = int(num_train * args.val_split)
x_val, y_val = x_train[-val_size:], y_train[-val_size:]
x_train, y_train = x_train[:-val_size], y_train[:-val_size]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000, seed=seed_value, reshuffle_each_iteration=True).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 2️⃣ Define Optimized CNN Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax', dtype='float32')  # ensure float32 output under mixed precision
])

# -----------------------------
# 3️⃣ Compile Model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 3.1️⃣ Callbacks
# -----------------------------
os.makedirs(os.path.dirname(args.model_h5) or '.', exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=args.model_h5, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

# -----------------------------
# 4️⃣ Train Model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    verbose=1,
    callbacks=callbacks
)

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"💾 Saving best model to '{args.model_h5}' and SavedModel to '{args.model_dir}'...")

# -----------------------------
# 6️⃣ Save Model
# -----------------------------
# Save latest weights in both formats; best H5 already handled by checkpoint
model.save(args.model_dir, include_optimizer=False)
model.save(args.model_h5)
print("✅ Models saved successfully!")

# -----------------------------
# 7️⃣ Optional: Plot Training Curves
# -----------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.close()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.close()
