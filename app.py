from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import cv2
import shutil
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="template")

# Load models once
reg = pickle.load(open("PCOS_FINAL_SEVERITY/model.pkl", "rb"))
reg1 = pickle.load(open("PCOS_FINAL_SEVERITY/model1.pkl", "rb"))

@app.route("/pcos")
def pcos():
    return render_template("userlog.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/choose")
def choose():
    return render_template("choose.html")

@app.route("/remedy")
def remedy():
    return render_template("sol.html")

@app.route("/test")
def test():
    return render_template("test.html")

@app.route("/test1")
def test1():
    return render_template("test2.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(request.form.get(chr(98 + i), 0)) for i in range(27)]  # Reads inputs from 'b' to 'za'
    arr = np.array([data])
    pred = reg.predict(arr)
    print(pred)
    return render_template("index.html", data=pred, d14=int(data[13]))

@app.route("/predict1", methods=["POST"])
def predict1():
    data = [float(request.form.get(chr(98 + i), 0)) for i in range(37)]  # Reads inputs from 'b' to 'zk'
    arr = np.array([data])
    pred1 = reg1.predict(arr)
    print(pred1)
    return render_template("index.html", data=pred1)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        dirPath = "static/images"
        
        # Clear previous images
        for fileName in os.listdir(dirPath):
            os.remove(os.path.join(dirPath, fileName))
        
        fileName = request.form['filename']
        dst = "static/images"
        shutil.copy(os.path.join("static/test", fileName), dst)
        
        image = cv2.imread(os.path.join("static/test", fileName))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        _, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)

        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'PCOS_DETECTION1-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_data = cv2.imread(path, cv2.IMREAD_COLOR)
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img_data), img])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()

        # Resetting the default graph to avoid conflicts
        tf.compat.v1.reset_default_graph()

        convnet = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
        convnet = tflearn.conv_2d(convnet, 32, 3, activation='relu')
        convnet = tflearn.max_pool_2d(convnet, 3)
        convnet = tflearn.conv_2d(convnet, 64, 3, activation='relu')
        convnet = tflearn.max_pool_2d(convnet, 3)
        convnet = tflearn.conv_2d(convnet, 128, 3, activation='relu')
        convnet = tflearn.max_pool_2d(convnet, 3)
        convnet = tflearn.fully_connected(convnet, 1024, activation='relu')
        convnet = tflearn.dropout(convnet, 0.8)
        convnet = tflearn.fully_connected(convnet, 2, activation='softmax')
        convnet = tflearn.regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('Model loaded!')

        # Visualization and prediction
        fig = plt.figure()
        str_label = ""
        accuracy = ""
        for num, data in enumerate(verify_data):
            img_data = data[0]
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]
            print(model_out)
            if np.argmax(model_out) == 0:
                str_label = "Normal"
                accuracy = "Normal with {:.2f}%".format(model_out[0] * 100)
            elif np.argmax(model_out) == 1:
                str_label = "PCOS"
                accuracy = "PCOS with {:.2f}%".format(model_out[1] * 100)

        return render_template('userlog.html', status=str_label, accuracy=accuracy,
                               ImageDisplay=url_for('static', filename='images/' + fileName),
                               ImageDisplay1=url_for('static', filename='gray.jpg'),
                               ImageDisplay2=url_for('static', filename='edges.jpg'),
                               ImageDisplay3=url_for('static', filename='threshold.jpg'))

    return render_template('userlog.html')

if __name__ == "__main__":
    app.run(debug=True)