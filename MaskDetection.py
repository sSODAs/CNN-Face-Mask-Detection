import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau    
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "archive\data"
CATEGORIES = ["with_mask", "without_mask"]  # Define labels

def preprocess_image(img):
    """ ปรับ Contrast และ Normalize สี """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)  # ปรับ Contrast เฉพาะ L channel

    # Merge back LAB
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  

    # 🔥 Normalize สีโดยแปลงเป็น HSV แล้วปรับค่า V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)  # Normalize ค่าแสงในภาพ
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # แปลงกลับเป็น BGR

    return img

def load_dataset(dataset_path, categories, img_size=(224, 224)):
    data, labels = [], []

    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        label = categories.index(category)  # Assign label (0 or 1)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            try:
                img = cv2.resize(img, img_size)  # ปรับขนาดภาพ
                img = preprocess_image(img)  # 🔥 ใช้ฟังก์ชัน preprocessing
                data.append(img)
                labels.append(label)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # Convert to NumPy arrays
    data = np.array(data) / 255.0  # Normalize ค่าพิกเซล
    labels = np.array(labels)

    return data, labels

data, labels = load_dataset(DATASET_PATH, CATEGORIES)
data, labels = shuffle(data, labels, random_state=42)

split_index = int(0.7 * len(data)) 
data_train, data_test = data[:split_index], data[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

mask_indices = np.where(labels_train == 0)[0]  # Indices of "With Mask"
extra_mask_images = data_train[mask_indices]
extra_mask_labels = labels_train[mask_indices]

# Append extra mask images to dataset
data_train = np.concatenate((data_train, extra_mask_images), axis=0)
labels_train = np.concatenate((labels_train, extra_mask_labels), axis=0)

# Shuffle dataset again
data_train, labels_train = shuffle(data_train, labels_train, random_state=42)

print(f"Train data: {data_train.shape}, Train labels: {labels_train.shape}")
print(f"Test data: {data_test.shape}, Test labels: {labels_test.shape}")

# 🔥 อัปเดต ImageDataGenerator (ลบ brightness_range ออก)
datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(data_train)

# Define the CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Regularization
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.0005)

# Compile model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

history = model.fit(datagen.flow(data_train, labels_train, batch_size=16), 
                    epochs=20,  
                    validation_data=(data_test, labels_test), 
                    callbacks=[lr_scheduler])

# Save model
model.save("mask_detection_model.h5")
print("Model saved successfully.")

# Load model
model = load_model("mask_detection_model.h5")
print("Model loaded successfully!")

# Evaluate model
test_loss, test_acc = model.evaluate(data_test, labels_test)
print(f"Test Accuracy: {test_acc:.2%}")

# Select 10 random test images
random_indices = random.sample(range(len(data_test)), 10)

plt.figure(figsize=(10, 5))

for i, idx in enumerate(random_indices):
    sample_img = np.expand_dims(data_test[idx], axis=0)  
    prediction = model.predict(sample_img)[0][0]  

    pred_label = "With Mask" if prediction < 0.5 else "Without Mask"
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(data_test[idx])
    plt.title(pred_label)
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

from sklearn.metrics import classification_report

preds = (model.predict(data_test) > 0.5).astype(int)  
print(classification_report(labels_test, preds, target_names=["With Mask", "Without Mask"]))
