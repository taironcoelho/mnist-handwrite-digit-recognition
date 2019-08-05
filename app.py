import numpy as np
import tensorflow as tf
import os

from keras.models import load_model
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__, template_folder='templates')
port = int(os.environ.get('PORT', 3000))

def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('model/model.h5')
    graph = tf.get_default_graph()

@app.route('/')
def render_index():
   return render_template('index.html')
	
@app.route('/predict', methods = ['POST'])
def predict_number():
    req_img = Image.open(request.files['file'].stream).convert("L")
    req_img = req_img.resize((28,28))
    img_arr = np.array(req_img)
    img_arr = img_arr.reshape(1,28,28,1)
    with graph.as_default():
        predicted_number = model.predict_classes(img_arr)

    
    return 'Number Predicted: ' + str(predicted_number[0])
		
if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=port)