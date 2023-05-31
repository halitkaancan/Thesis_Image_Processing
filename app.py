import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

@app.route('/process_photo')
def hello():
    return 'Hello from Flask server!!'

@app.route('/', methods=['POST'])
def process_photo():
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 255]
    colors = [(0,0,255), (0,51,255), (0,102,255), (0,153,255), (0,204,255), (0,255,255), (102,255,178), (102,255,102), (204,255,102), (255,255,102), (255,204,102), (255,153,102), (255,102,102), (255,51,102), (255,0,102), (204,0,204), (153,0,204), (102,0,204), (51,0,204), (0,0,204), (0,51,204), (0,102,204), (0,153,204), (0,204,204), (0,255,204)]

    def rescale(frame, scale=0.4):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        boyut = (width, height)
        return cv2.resize(frame, boyut, interpolation=cv2.INTER_AREA)

    file = request.files['photo']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = rescale(img)

    intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(intensity, thresholds[16], 255, cv2.THRESH_TOZERO)

    data = img[(thresholded>0)&(thresholds[18]>intensity)]

    coords = np.argwhere(thresholded>0)

    coordinates = []
    for (y, x), (b, g, r) in zip(coords, data):
        coordinates.append({'x': int(x), 'y': int(y), 'b': int(b), 'g': int(g), 'r': int(r)})

    result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    for i in range(len(thresholds) - 1):
        result[(intensity >= thresholds[i]) & (intensity < thresholds[i+1])] = colors[i]

    _, encoded_result = cv2.imencode('.png', result)
    result_data = encoded_result.tobytes()
    result_data_base64 = base64.b64encode(result_data).decode('utf-8')

    return jsonify({'result': result_data_base64})

@app.route('/', methods=['GET'])
def hello_ui():
    return 'Hello from UI!'

if __name__ == '__main__':
    app.run(debug=True)
