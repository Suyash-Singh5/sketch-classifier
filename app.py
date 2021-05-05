from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import keras
import tensorflow as tf
from tensorflow.python.keras.backend import set_session



from PIL import Image
from io import BytesIO
from base64 import b64decode
import numpy as np
import cv2
import joblib

app = Flask(__name__)


model = models.Sequential()

model.add(Conv2D(32, 3, padding="same", input_shape=(160, 160, 1)))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, 3, padding="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, 3, padding="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, padding="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.75))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation="softmax"))

opt = Adam(learning_rate=0.003)
model.compile(opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("CNNVadapav.ckpt")


def get_model_prediction():
    temp_img = cv2.imread("temp.png", cv2.IMREAD_UNCHANGED)
    cv_img = cv2.cvtColor(np.float32(np.array(temp_img)), cv2.COLOR_RGB2BGRA)
    trans_mask = cv_img[:, :, 3] == 0
    cv_img[trans_mask] = [255, 255, 255, 255]
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_img = abs(255 - cv_img)
    cv_img = cv2.resize(cv_img, (160, 160))
    cvg = cv_img
    cv_img = cv_img / 255.0
    cv_img = np.reshape(cv_img, (160, 160, 1))
    cv_img = np.array([cv_img])
    ans = np.argmax(model.predict(cv_img)[0])
    dic_cool = joblib.load("vadapav.joblib")
    ans = list(dic_cool.keys())[(list(dic_cool.values())).index(ans)]
    return ans


@app.route('/')
def hello_world():
    return 'This is Home Page.'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        data = json.loads(request.data)
        imagestr = str(data['imagestring'])
        im = Image.open(BytesIO(b64decode(imagestr.split(',')[1])))
        im.save("temp.png")
        ans = get_model_prediction()
        print(ans)
        response = jsonify({
            "predictedclass": ans,
        })
        return response


if __name__ == "__main__":
    app.run(port=3333, debug=True)
