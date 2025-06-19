import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#Load and Process all Images
def load_images(folder, label, image_size=(64,64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder,filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.flatten())
            labels.append(label)
    return images,labels


cat_train_images, cat_train_labels = load_images('./content/Train/cat', 'cat')
dog_train_images, dog_train_labels = load_images('./content/Train/dog', 'dog')

x_train = np.array(cat_train_images + dog_train_images)
y_train = np.array(cat_train_labels + dog_train_labels)

cat_test_images, cat_test_labels = load_images('./content/Test/cat', 'cat')
dog_test_images, dog_test_labels = load_images('./content/Test/dog', 'dog')

x_test = np.array(cat_test_images + dog_test_images)
y_test = np.array(cat_test_labels + dog_test_labels)

x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"Loaded {len(x_train)} training samples.")
# print("Sample train labels:", y_train[:5])
print(f"Loaded {len(x_test)} test samples.")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

svm_model = LinearSVC(max_iter=5000)
print("Fitting the model....")
svm_model.fit(x_train, y_train_enc)
print(" Model trained successfully.!")

# from inspect import classify_class_attrs
y_pred = svm_model.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))
print('\n Classification Report:')
print(classification_report(y_test_enc, y_pred, target_names = le.classes_))


def predict_image(path):
  img = cv2.imread(path)
  img = cv2.resize(img, (64,64))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_flat = img.flatten()
  pred = svm_model.predict([img_flat])
  plt.imshow(img, cmap='gray')
  plt.title(f'Prediction: {le.inverse_transform(pred)[0]}')
  plt.axis('off')
  plt.show()

predict_image('./content/Test_images/image4.jpg')

