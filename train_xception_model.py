import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paths
data_dir = 'C:\\Users\\shiva\\OneDrive\\Desktop\\DeepFake Detection(2)\\dataset\\processed'

# Data Preprocessing
img_size = (299, 299)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Model Building
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Optional: Unfreeze for fine-tuning
# for layer in base_model.layers:
#     layer.trainable = True
# model.compile(...)
# model.fit(...)

# Save model
model.save("xception_deepfake_model.h5")

# Evaluation
val_gen.reset()
preds = model.predict(val_gen)
y_pred = (preds > 0.5).astype(int)
y_true = val_gen.classes

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Plotting
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')       
plt.title("Model Loss")
plt.legend()
plt.show()

# end of file
# This script trains a deep learning model using the Xception architecture to classify images as real or fake.      
# It preprocesses the images, builds the model, trains it, evaluates its performance, and plots the training history.
# The script uses Keras for model building and training, and TensorFlow for backend operations.
# The script uses the ImageDataGenerator class to preprocess the images, including rescaling and splitting the data into training and validation sets.
# The model is built using the Xception architecture, which is a deep convolutional neural network designed for image classification tasks.
# The model is compiled with the Adam optimizer and binary crossentropy loss function, suitable for binary classification tasks.
# The model is trained for 5 epochs, and the training history is saved for later analysis.