
# 🐱🐶 Cat vs Dog Image Classification using LinearSVC

This project implements a **Support Vector Machine (SVM)** classifier using `LinearSVC` from `scikit-learn` to classify images of **cats** and **dogs** from the Kaggle Dogs vs Cats dataset. Images are preprocessed (grayscale, resized, flattened) and fed into a classical machine learning model for binary classification.

---

## 📁 Dataset Overview

- **Source**: Kaggle – Dogs vs Cats Dataset  
- **Train Path**: `./content/Train/`  
- **Test Path**: `./content/Test/`  
- **Prediction Samples**: `./content/Test_images/`  
- **Categories**: `cat`, `dog`  
- **Image Size**: Resized to 64×64 (grayscale)

---

## 📌 Problem Statement

Build a binary image classifier that can predict whether a given image is of a **cat** or a **dog** using traditional ML techniques, specifically **Support Vector Machine (SVM)** with linear kernel.

---

## 🚀 How It Works

### 1. Importing Required Libraries

- `opencv-python` for image loading and processing  
- `numpy` for numerical operations  
- `scikit-learn` for training and evaluation  
- `matplotlib` for visualization

### 2. Image Loading & Preprocessing

- Convert to grayscale  
- Resize to 64x64 pixels  
- Flatten into 1D vector  
- Normalize pixel values (0–1 range)

### 3. Label Encoding

Encode `'cat'` and `'dog'` labels into numeric values using `LabelEncoder`.

### 4. Model Training

Train a **LinearSVC** model on the processed images:

```python
svm_model = LinearSVC(max_iter=5000)
svm_model.fit(x_train, y_train_enc)
```

### 5. Evaluation

Evaluate model using:
- Confusion Matrix
- Classification Report

### 6. Prediction Demo

You can test the model on any image using:

```python
predict_image('./content/Test_images/image4.jpg')
```

It displays the image with predicted label.

---

## 📎 Requirements

Install dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

---

## 📈 Sample Output

```
Loaded 8000 training samples.
Loaded 2000 test samples.

Fitting the model....
Model trained successfully.!

Confusion Matrix:
[[950  50]
 [ 45 955]]

Classification Report:
              precision    recall  f1-score   support

         cat       0.95      0.95      0.95      1000
         dog       0.95      0.96      0.96      1000

    accuracy                           0.95      2000
```

---

## 📚 Learnings

- Classical ML application for image classification
- Data preprocessing using OpenCV
- Training & evaluating SVM using `scikit-learn`
- Visualizing prediction output with Matplotlib

---

## 🧠 Future Improvements

- Apply PCA to reduce dimensionality  
- Try kernel SVM (`rbf`, `poly`)  
- Build an interactive UI using **Streamlit**  
- Switch to **CNN (Convolutional Neural Networks)** for deeper learning

---

## 👨‍💻 Author

**Yash Patel**  
Aspiring Machine Learning Engineer | Python Developer | AI Explorer
