from flask import Flask, request, Response, send_file
import yolo
import io
from PIL import Image
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from os import remove
app = Flask(__name__)

yolo.YOLO._defaults['model_path']='model_data/yolo-openimages.h5'
yolo.YOLO._defaults['classes_path']='model_data/openimages.names'
yolo.YOLO._defaults['anchors_path']='model_data/yolo_anchors.txt'
model = yolo.YOLO()

@app.route('/isAlive', methods=['POST', 'GET'])
def index():
    return "true"
    
@app.route('/prediction/image', methods=['POST'])
def image_prediction():
    file = request.files['file']
    try:
        image = Image.open(file)
    except:
        return str("Image Error")
    else:
        image = model.detect_image(image)
        image.save("output.jpg")
        return send_file("output.jpg",mimetype='image/jpeg')
   
if __name__ == '__main__':
    app.run(host='0.0.0.0')    
# app.run(debug=True)