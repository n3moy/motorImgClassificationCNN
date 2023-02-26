import os
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import torch
from flask import Blueprint, request, render_template
from ...models.model import CNN
from ...data.img_handler import ImgHandler


mod = Blueprint('backend', __name__, template_folder='templates', static_folder='./static')

@mod.route('/')
def home():
    return render_template('index.html')

@mod.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
           return "Someting went wrong : file not found"
        
        input_file = request.files['file']
        if input_file.filename == '':
            return "No input file" 
        else:
            path = os.path.join('modules/inputs/'+input_file.filename)
            input_file.save(path)
            prediction = identify_image(path)
            return prediction

def identify_image(img_path):
    config_path = Path(__file__).resolve().parents[3] / 'configs' / 'model' / 'inference.yaml'
    config = yaml.load(open(config_path), Loader=SafeLoader)
    config['image_path'] = img_path
    handler = ImgHandler(config)
    instance = handler.prepare(mode='inference')
    model_cnn = CNN(img_size=None, out_size=11)
    model_path = config['model_path_web']

    model_cnn.load_state_dict(torch.load(model_path))
    model_cnn.eval()
    pred = model_cnn(instance)
    pred = config['mapping'][torch.argmax(pred)]
    return pred

