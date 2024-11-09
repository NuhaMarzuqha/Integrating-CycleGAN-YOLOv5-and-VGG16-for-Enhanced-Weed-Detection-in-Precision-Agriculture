### *Synthetic Data Augmentation and Deep Learning for Real-Time Weed Detection in Agricultural Fields

## Overview

This project aims to develop an advanced system for real-time weed detection in agricultural fields. By integrating YOLOv5 for object detection and VGG16 for refined classification, we aim to improve the accuracy and efficiency of identifying weeds in the presence of crops like potatoes and carrots. The system utilizes a custom dataset, and CycleGAN is employed to generate synthetic, clean images of carrots from carrot-and-weed composites, addressing data limitations and supporting generalization across various crop environments.

## Problem Statement

Effective weed management is a critical challenge in modern agriculture. The traditional approach of manual weeding or chemical herbicide application can be inefficient and harmful to the environment. Our solution leverages machine learning techniques for precision agriculture to automate the detection and classification of weeds in agricultural fields, ensuring better crop management and reduced herbicide use.

## Approach

1. **YOLOv5 for Object Detection**: YOLOv5 is used for real-time localization of crops and weeds in images. It generates bounding boxes around the detected objects, providing regions of interest for further analysis.

2. **VGG16 for Classification**: Once the objects (crops and weeds) are localized, VGG16, a convolutional neural network, is applied to classify the regions. This step enhances the model’s ability to differentiate between crops and weeds with high accuracy.

3. **CycleGAN for Synthetic Data Augmentation**: To address data limitations, particularly for clean crop images, a CycleGAN model is employed to generate synthetic carrot images from composite carrot-and-weed images. This synthetic data augmentation helps expand the dataset, improving model performance and generalizability across different environments.

## Results

Our combined pipeline shows significant improvements in precision and recall when compared to single-model systems. The modular integration of YOLOv5 and VGG16, along with the CycleGAN-based synthetic data augmentation, provides a robust solution for real-time weed detection in field-based environments.

## Features

- **Real-Time Detection**: Using YOLOv5 for fast and accurate localization of weeds and crops.
- **High Classification Accuracy**: VGG16 ensures precise classification of crops and weeds after localization.
- **Synthetic Data Generation**: CycleGAN enhances model training by generating additional data, improving model robustness.
- **Modular and Scalable**: The system is designed to be modular, allowing easy integration with other agricultural systems.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:

    ```bash
    python main.py
    ```

## Dependencies

- Python 3.x
- YOLOv5
- VGG16 (TensorFlow/Keras)
- CycleGAN (TensorFlow)
- OpenCV
- NumPy
- Matplotlib

## Evaluation Metrics

- **Precision**: Measures the accuracy of the positive predictions made by the model.
- **Recall**: Measures the ability of the model to correctly identify all positive instances (weeds).
- **F1-Score**: The harmonic mean of precision and recall.

## Conclusion

This system offers a promising solution for automating weed detection in agricultural fields, contributing to the broader goals of precision agriculture. The combination of YOLOv5 for detection, VGG16 for classification, and CycleGAN for synthetic data generation provides a robust framework for real-time monitoring and crop management.

## Authors

- Tshewang Rigzin
- M I Nuha Marzuqha
- Sonam Tharchen  
Vellore Institute of Technology University

## Acknowledgements

We would like to thank the Vellore Institute of Technology for their support and resources provided during this research.

---

Feel free to modify or add sections based on your needs!
