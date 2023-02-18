from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten

app = Flask(__name__)

# Define the dataset directory
dataset_dir = "dataset"

# Define the image size
img_size = 128

# Define the classes
classes = ["healthy", "diseased"]

# Load the model
model = load_model('model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_size, img_size))
    return np.array([image])

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for the predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the form
    file = request.files['image']

    # Read the image and preprocess it
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = preprocess_image(image)

    #Use the model to make a prediction
    prediction = model.predict(image.reshape(-1, img_size, img_size, 1))
    class_index = np.argmax(prediction)
    disease_name = classes[class_index]

    # Render the result page with the prediction
    return render_template('result.html', prediction=disease_name)

if __name__ == '__main__':
    app.run(debug=True)


