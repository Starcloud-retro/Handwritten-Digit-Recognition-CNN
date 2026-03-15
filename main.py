"""
AI-Based Handwritten Digit Recognition System
============================================
Full pipeline: Load MNIST → Build CNN → Train → Evaluate → Predict
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Sklearn for evaluation
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Pillow for custom image prediction
from PIL import Image, ImageOps

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS MNIST
# ─────────────────────────────────────────────

def load_and_preprocess():
    """Load MNIST, normalize, reshape, one-hot encode."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values: 0-255 → 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    # Reshape for CNN: (samples, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    print(f"Training samples : {X_train.shape[0]}")
    print(f"Test samples     : {X_test.shape[0]}")
    print(f"Image shape      : {X_train.shape[1:]}")

    return X_train, y_train, X_test, y_test, y_train_cat, y_test_cat


# ─────────────────────────────────────────────
# 2. BUILD CNN MODEL
# ─────────────────────────────────────────────

def build_cnn():
    """
    CNN Architecture:
    Conv2D(32) → Conv2D(64) → MaxPool → Dropout
    → Flatten → Dense(128) → Dropout → Dense(10, softmax)
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Fully Connected
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')   # 10 digit classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
# 3. BUILD MLP MODEL (optional comparison)
# ─────────────────────────────────────────────

def build_mlp():
    """Simple MLP for comparison."""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ─────────────────────────────────────────────
# 4. TRAIN THE MODEL
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train_cat, X_test, y_test_cat,
                epochs=15, batch_size=128, save_path="best_model.keras"):
    """Train with early stopping and model checkpointing."""
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────
# 5. EVALUATE & VISUALIZE
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, y_test_cat, save_dir="outputs"):
    """Accuracy, loss, confusion matrix, classification report."""
    os.makedirs(save_dir, exist_ok=True)

    # Overall metrics
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy : {test_acc*100:.2f}%")
    print(f"   Test Loss     : {test_loss:.4f}")

    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.show()
    print(f"Saved: {save_dir}/confusion_matrix.png")

    return y_pred, y_pred_prob


def plot_training_history(history, save_dir="outputs"):
    """Plot accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy',      color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss',      color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.show()
    print(f"Saved: {save_dir}/training_curves.png")


def visualize_predictions(model, X_test, y_test, n=25, save_dir="outputs"):
    """Show sample predictions with colour-coded correctness."""
    os.makedirs(save_dir, exist_ok=True)

    indices   = np.random.choice(len(X_test), n, replace=False)
    y_pred    = np.argmax(model.predict(X_test[indices]), axis=1)

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[indices[i]].reshape(28, 28), cmap='gray')
        correct  = y_pred[i] == y_test[indices[i]]
        color    = 'green' if correct else 'red'
        ax.set_title(f"P:{y_pred[i]}  T:{y_test[indices[i]]}", color=color, fontsize=9)
        ax.axis('off')

    plt.suptitle('Sample Predictions  (Green=Correct, Red=Wrong)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/sample_predictions.png")


def show_errors(model, X_test, y_test, n=20, save_dir="outputs"):
    """Show misclassified images."""
    os.makedirs(save_dir, exist_ok=True)

    y_pred     = np.argmax(model.predict(X_test), axis=1)
    wrong_idx  = np.where(y_pred != y_test)[0]

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i >= min(n, len(wrong_idx)):
            ax.axis('off'); continue
        idx = wrong_idx[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"Pred:{y_pred[idx]}  True:{y_test[idx]}", color='red', fontsize=9)
        ax.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/misclassified.png", dpi=150)
    plt.show()
    print(f"Saved: {save_dir}/misclassified.png")


# ─────────────────────────────────────────────
# 6. CUSTOM IMAGE PREDICTION
# ─────────────────────────────────────────────

def predict_custom_image(model, image_path):
    """
    Predict digit from a custom image file.
    Accepts any image; auto-resizes to 28x28 grayscale.
    """
    img = Image.open(image_path).convert("L")   # grayscale
    img = ImageOps.invert(img)                   # MNIST: white digit on black
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    probs     = model.predict(arr)[0]
    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100

    print(f"\n🔢 Predicted Digit : {predicted}")
    print(f"   Confidence      : {confidence:.2f}%")
    print(f"   All scores      : {dict(enumerate(probs.round(3)))}")

    plt.figure(figsize=(5, 5))
    plt.imshow(arr.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted}  ({confidence:.1f}% confidence)", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return predicted, confidence


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI-Based Handwritten Digit Recognition System")
    print("=" * 60)

    # Step 1 – Data
    X_train, y_train, X_test, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    # Step 2 – Model
    cnn_model = build_cnn()

    # Step 3 – Train
    history = train_model(cnn_model, X_train, y_train_cat, X_test, y_test_cat,
                          epochs=15, batch_size=128, save_path="best_cnn_model.keras")

    # Step 4 – Evaluate
    plot_training_history(history)
    evaluate_model(cnn_model, X_test, y_test, y_test_cat)
    visualize_predictions(cnn_model, X_test, y_test)
    show_errors(cnn_model, X_test, y_test)

    # Step 5 – Optional: compare with MLP
    print("\n--- Training MLP for comparison ---")
    mlp_model   = build_mlp()
    mlp_history = train_model(mlp_model, X_train, y_train_cat, X_test, y_test_cat,
                               epochs=10, batch_size=128, save_path="best_mlp_model.keras")
    mlp_loss, mlp_acc = mlp_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nCNN Accuracy : {cnn_model.evaluate(X_test, y_test_cat, verbose=0)[1]*100:.2f}%")
    print(f"MLP Accuracy : {mlp_acc*100:.2f}%")

    # Step 6 – Custom image (uncomment and set path)
    # predict_custom_image(cnn_model, "my_digit.png")

    print("\n✅ All done! Check the 'outputs/' folder for saved plots.")
