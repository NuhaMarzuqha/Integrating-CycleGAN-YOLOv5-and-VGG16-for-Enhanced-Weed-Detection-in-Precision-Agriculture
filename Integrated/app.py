from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from integrate import process_image

app = Flask(__name__)

# Configure the upload and output folders
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the image with YOLO and VGG16
        yolo_result, vgg_results = process_image(file_path)
        
        # Convert YOLO result path to URL for the HTML template
        yolo_result_url = url_for('output_file', filename=os.path.basename(yolo_result))
        
        # Convert VGG16 results (with paths and labels) to URLs and format for HTML template
        vgg_result_urls = [{"path": url_for('output_file', filename=os.path.basename(result["path"])), 
                            "label": result["label"]} for result in vgg_results]
        
        # Render the results on the page
        return render_template('index.html', yolo_result=yolo_result_url, vgg_results=vgg_result_urls)

# Route to serve the static images
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
