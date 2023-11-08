from flask import Flask, jsonify, request 

import os
# from nudenet import NudeDetector
from nudenet import NudeClassifier
from tf.classifier_customized import Classifier
from tf.detect import NudeDetector

# from django.http import HttpResponse  
import urllib.request
  
# creating a Flask app 
app = Flask(__name__) 
@app.route('/', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'): 
        data = "Go to /predictor/<YOUR IMAGE URL EXP: '/predictor/https://img.freepik.com/premium-photo/portrait-beautiful-sexy-girl-with-perfect-body_942478-2138.jpg'> "
        return jsonify({'data': data}) 
@app.route('/predictor/', methods = ['GET']) 
def predict(): 
    url = request.args.get('url')
    print(url)
    api = url
    api_urls = []
    
    api_urls = [api]
    
    # api_urls = ['https://media.wired.com/photos/593261cab8eb31692072f129/master/w_2560%2Cc_limit/85120553.jpg'] 

    for api in api_urls:
        nude_detector = NudeDetector()
        # nude_classifier = NudeClassifier()
        nude_classifier = Classifier()
        # classifier = my_classification()
        directory, filename = os.path.split(api)
        file_path = os.path.join("media", filename)
        # urllib.request.urlretrieve(api, file_path)
        urllib.request.urlretrieve(api, file_path)
        path = "D:/School Work/maxx_ai/media/"
        #Nude Detection
        detections = nude_detector.detect(path + filename)
        #Nude Classifier
        # images_preds = nude_classifier.classify(path + filename)
        images_preds = nude_classifier.classify(path + filename)

        
        # # images_preds = nude_detector.censor(path + filename)
        # # detections = nude_detector.censor(path + filename)

    return jsonify({'Object Detection': detections, 'NSFW Classifier': images_preds})
    
if __name__ == '__main__': 
  
  # run app in debug mode on port 5000
    app.run(debug = True, port=5000, host='0.0.0.0')
    # app.run(debug = True) 