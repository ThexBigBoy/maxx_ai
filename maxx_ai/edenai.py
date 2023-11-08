import json
from flask import Flask, jsonify, request 
import os
from nudenet import NudeDetector
import urllib.request
import requests

app = Flask(__name__) 

@app.route('/', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'): 
  
        data = "hello world"
        return jsonify({'data': data}) 

@app.route('/test/', methods = ['GET']) 
def disp(): 
    url = request.args.get('url')
    api = url
    api_urls = []
    api_urls = [api]
    for api in api_urls:
        print(api)
        # nude_detector = NudeDetector()
        directory, filename = os.path.split(api)
        file_path = os.path.join("media", filename)
        print(filename)
        urllib.request.urlretrieve(api, file_path)
        path = "D:/School Work/maxx_ai/media/"

    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNDdhMTc5MjUtZTcwMi00ZWY2LTk2OWYtY2Q4NDZkNThkMGM1IiwidHlwZSI6ImFwaV90b2tlbiJ9.B8EfIAlZaFBUMDYJ289mrq7liFzXl9tqv8q_Ouzo4mo"}
    url = "https://api.edenai.run/v2/image/object_detection"
    data = {"providers" :"google, amazon"}
    files = {'file': open(path+filename,'rb')}
    response = requests.post(url, data=data, files=files, headers=headers)
    result = json.loads(response.text)

    return jsonify({'data': result})
    
if __name__ == '__main__': 
    app.run(debug = True, port=5000, host='0.0.0.0')