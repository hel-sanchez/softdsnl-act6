import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image
import os

# Paths
train_dir = "data/training_set"
test_dir = "data/testing_set"

print(f"Training directory: {train_dir}")
print(f"Testing directory: {test_dir}")

# Image generators for loading dataset
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
print(f"Number of training samples: {train_generator.samples}")
print(f"Training classes: {train_generator.class_indices}")

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
print(f"Number of validation samples: {val_generator.samples}")

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Model compiled.")

# Train model
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    verbose=2  # Minimal per-epoch output
)
print("Training finished.")

# Save model
model_path = "kaggle_cnn_model.h5"
model.save(model_path)
print(f"Model saved as {model_path}")

# Evaluate on test set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
print(f"Number of test samples: {test_generator.samples}")

print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
