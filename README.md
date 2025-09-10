# 🧠 SOFTDSNL Activity 6: Custom Image Classification with Kaggle Dataset

## 📌 Overview
For your midterm project, you will build a **custom image classification model** using **any dataset from Kaggle**.  
This project extends what we did with MNIST and CIFAR-10, but now you are free to explore real-world datasets.

You will also **deploy your trained model in Django**, so that it accepts image uploads (via Postman) and responds with predictions.

---

## 🎯 Learning Objectives
By completing this project, you will:
- Learn how to source and prepare datasets from Kaggle.
- Train and evaluate a CNN model for image classification.
- Connect your trained model to a Django backend.
- Test predictions via Postman.

---

## File Structure

kaggle_image_classification/
│── requirements.txt
│── README.md
│── train_model.py
│── test_model.py
│── data/              # extracted dataset from Kaggle
│── my_test_images/    # your own test images for predictions
│── ml_api_project/    # Django project folder

---

## 📂 Deliverables
Each student must submit:

1. ✅ A **PDF report** named `SOFTDSNL_Activity6_Surnames.pdf` that includes:
   - Screenshots of model training results (accuracy/loss curves).
   - Screenshots of **10 Postman test cases** (1 image per category).
   - A **link to your GitHub repository fork** containing your code.

2. ✅ A **GitHub repository** with:
   - Training script (Jupyter notebook or `.py`).
   - Trained model file (`.h5`).
   - Django project with an endpoint for predictions.
   - README with setup instructions.

---

## 📝 Instructions

### 1. Choose a Kaggle Dataset
- Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and pick an image classification dataset.  
- Examples: 
  - Cats vs Dogs
  - Handwritten Letters
  - Fruits/Vegetables classification
- Download the dataset and place it in your project directory.

---

### 2. Preprocess the Data
- Load and normalize images.
- Resize all images to a fixed size (e.g., 64x64 or 128x128).
- Split into training and testing sets.

---

### 3. Build and Train a CNN
- Use TensorFlow/Keras to define a CNN model.
- Compile, train, and evaluate the model.
- Save your trained model as `.h5`.

```
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Paths
train_dir = "data/train"
test_dir = "data/test"

# Image generators for loading dataset
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

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

# Train model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# Save model
model.save("kaggle_cnn_model.h5")
print("Model saved as kaggle_cnn_model.h5")
```

---

### 4. Deploy with Django
- Create a Django project.
- Add an endpoint (e.g., `/predict/`) that:
  1. Accepts an image file upload.
  2. Preprocesses the image (resize, normalize).
  3. Loads your trained model.
  4. Returns the predicted class in JSON.

---

### 5. Test with Postman
- Send **10 test requests** (one per category in your dataset).
- Take screenshots of successful predictions.

---

## 📊 Grading Criteria

| Criteria | Points |
|----------|---------|
| **PDF Report** | 30 |
| - Training results screenshots | 10 |
| - Postman test screenshots (10) | 10 |
| - GitHub link included | 10 |
| **Model Training** | 25 |
| - Correct dataset preprocessing | 10 |
| - CNN model architecture | 10 |
| - Accuracy/loss explanation | 5 |
| **Django Integration** | 25 |
| - Working endpoint for predictions | 15 |
| - Correct preprocessing in API | 10 |
| **Code Quality & Repo** | 20 |
| - Organized, readable code | 10 |
| - Clear README instructions | 10 |

**Total: 100 points**

---

## 🚀 Submission
- Upload your PDF report to the LMS.
- Include your GitHub repo link inside the PDF.
- Ensure your repo is public or shared with the instructor.

