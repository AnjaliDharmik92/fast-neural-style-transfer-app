from flask import Flask, render_template, request, jsonify
from openvino.inference_engine import IECore
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)
ie = IECore()

# Load the OpenVINO model
model_xml = "/models/1/fast-neural-style-mosaic.xml"
model_bin = "/models/1/fast-neural-style-mosaic.bin"
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name='CPU')

# Ensure 'uploads' directory exists
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

# Specify the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')
    
@app.route('/output')
def output_page():
    return render_template('output.html')

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform style transfer
        result_path = perform_style_transfer(file_path)
        
        # Get the URLs for original and stylized images
        original_image_url = f'/uploads/{filename}'
        stylized_image_url = '/static/stylized/result.jpg'  # Adjust this based on your actual stylized image path

        return jsonify({'original_image': original_image_url, 'stylized_image': stylized_image_url})
    else:
        return jsonify({'error': 'Invalid file format'})


def perform_style_transfer(input_path):
    # Load the image and resize it to (224, 224)
    original_image = cv2.imread(input_path)
    original_image = cv2.resize(original_image, (224, 224))

    # Transpose the image to match the model's input format
    input_blob = next(iter(net.input_info))
    input_data = {input_blob: original_image.transpose((2, 0, 1)).reshape(1, 3, 224, 224)}

    # Perform style transfer
    result = exec_net.infer(inputs=input_data)

    # Transpose and reshape the output to match the image dimensions
    stylized_image = np.squeeze(result[next(iter(result))])
    stylized_image = np.transpose(stylized_image, (1, 2, 0))
    stylized_image = cv2.resize(stylized_image, (original_image.shape[1], original_image.shape[0]))

    # Save the stylized image
    stylized_path = os.path.join('static', 'stylized', 'result.jpg')
    cv2.imwrite(stylized_path, stylized_image)

    return stylized_path

if __name__ == '__main__':
    app.run(debug=True)
