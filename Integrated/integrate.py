import torch
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

# Paths
yolo_model_path = r'C:\Users\tshew\OneDrive\Desktop\yolov5\runs\train\continued_training2\weights\best.pt'
vgg16_model_path = r'C:\Users\tshew\OneDrive\Desktop\VGG16\vgg16_weed_detection.keras'
output_folder = r'C:\Users\tshew\OneDrive\Desktop\Integrated\output'

# Load YOLO model
yolo_model = torch.hub.load(r'C:\Users\tshew\OneDrive\Desktop\yolov5', 'custom', path=yolo_model_path, source='local')

# Load VGG16 model
vgg16_model = tf.keras.models.load_model(vgg16_model_path)
class_labels = ['carrot', 'potato', 'weed']

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to classify cropped image using VGG16
def classify_with_vgg16(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = vgg16_model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence

# Main function to process an image
def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Run YOLO detection
    results = yolo_model(image_path)
    
    # Extract YOLO result image with bounding boxes
    yolo_result_img = results.render()[0]
    yolo_output_path = os.path.join(output_folder, "yolo_result.png")
    cv2.imwrite(yolo_output_path, cv2.cvtColor(yolo_result_img, cv2.COLOR_RGB2BGR))

    # Process each detected object
    detections = results.pred[0]
    vgg_output_paths = []

    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = map(int, det[:6])  # Coordinates and class
        
        # Crop detected object
        crop_img = img.crop((x1, y1, x2, y2))
        
        # Classify with VGG16
        label, confidence = classify_with_vgg16(crop_img)
        
        # Save cropped image with classification
        crop_output_path = os.path.join(output_folder, f"crop_{i}_{label}.png")
        crop_img.save(crop_output_path)
        vgg_output_paths.append({"path": crop_output_path, "label": f"{label} ({confidence:.2f})"})
    
    return yolo_output_path, vgg_output_paths
