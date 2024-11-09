import subprocess

# Define paths and parameters
weights_path = "runs/train/continued_training2/weights/best.pt"  # Path to your trained weights
source_path = "data/images/train/file_160.png"  # Path to the image you want to test
img_size = "640"  # Image size for inference
conf_thresh = "0.25"  # Confidence threshold
save_name = "new_image_test"  # Directory name for saving results

# Define the command to call detect.py
command = [
    "python", "detect.py",
    "--weights", weights_path,
    "--source", source_path,
    "--img", img_size,
    "--conf", conf_thresh,
    "--name", save_name
]

# Run the command
subprocess.run(command)
