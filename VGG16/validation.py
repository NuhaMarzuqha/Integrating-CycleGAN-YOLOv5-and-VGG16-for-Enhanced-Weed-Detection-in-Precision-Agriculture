import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your saved model
model_path = r'C:\Users\tshew\OneDrive\Desktop\VGG16\vgg16_weed_detection.keras'
model = tf.keras.models.load_model(model_path)

# Path to the validation dataset
val_dir = r'C:\Users\tshew\OneDrive\Desktop\VGG16\val'

# Set up ImageDataGenerator for validation data
val_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize images

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,  # Adjust batch size as needed
    class_mode='categorical',
    shuffle=False  # No need to shuffle for validation
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
