import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Define the path to your saved model
model_path = r'C:\Users\tshew\OneDrive\Desktop\VGG16\vgg16_weed_detection.keras'
model = tf.keras.models.load_model(model_path)

# Define class labels (ensure they are in the correct order)
class_labels = ['carrot', 'potato', 'weed']

# Directory where you will put images to test
test_images_dir = r'C:\Users\tshew\OneDrive\Desktop\VGG16\test'

# Function to predict and display results
def predict_and_display(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize to match input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display the image with predicted label
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

# Loop through test images and make predictions
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(test_images_dir, filename)
        print(f"Processing {filename}...")
        predict_and_display(file_path)
