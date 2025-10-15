#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.python.keras.layers.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
os.environ='TF_ENABLE_ONEDNN_OPTS=0'

# In[2]:


# Path to your training and testing directories
train_data_dir = r'C:/Downloads/train'

# List of class labels (subdirectories) in the training data directory
class_labels = os.listdir(train_data_dir)

# Display random images from the training dataset
plt.figure(figsize=(40,40))
for i, class_label in enumerate(class_labels):
    class_path = os.path.join(train_data_dir, class_label)
    img_names = os.listdir(class_path)
    random_img_name = random.choice(img_names)
    random_img_path = os.path.join(class_path, random_img_name)
    img = load_img(random_img_path, target_size=(420, 420))  # Adjust target_size if needed
    plt.subplot(2, len(class_labels), i + 1)
    plt.imshow(img)
    plt.title(f'Training: {class_label}')
    plt.axis('off')


# # VGG16 | CNN MODEL

# In[3]:


from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam


# In[4]:


train_data_dir = "C:/Downloads/train"
test_data_dir = "C:/Downloads/test"


# In[5]:


image_height = 150
image_width = 150
batch_size = 32
num_classes = 10
epochs = 10


# In[6]:


# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[7]:


train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# In[8]:


# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))


# In[9]:


# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False


# In[10]:


# Build the model on top of VGG16
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


# In[11]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
)


# In[13]:


# Model evaluation
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)


# In[14]:


# Plot training and testing accuracy
plt.figure(figsize=(10, 6))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# In[15]:


# Predict test data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes


# In[16]:


# Accuracy metrics
accuracy = test_accuracy
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# In[17]:


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[18]:


#saving the model using it along with opencv 
from tensorflow.python.keras.models import load_model
# Save the model to a file
model.save("gesture_model_1.h5") #name of the modelwhat


# In[ ]:




