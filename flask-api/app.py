#app.py
import os
import requests
import json
import cv2
import numpy as np
import pickle
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = f"C:\\Users\rshuklaa\Documents\portfolio\dog_cats\api\static" ## Customize it

def predict(image_path,url ):

    img = Image.open(image_path)
    img = img.resize( (150, 150),resample=Image.BILINEAR).convert("RGB")   
    headers = {
      'Content-Type': 'application/json'
    }
    data = str(np.array(img).tolist())
    # print(f"data - {data}")
    response = requests.request("POST", url,headers=headers,  data=data)
    output = json.dumps(response.text)
    print(f"output -  {output}")

    prediction = json.loads(response.text)["body"]
    print(f"response -  {prediction}")
    return prediction


@app.route("/",methods = ["GET", "POST"])
def upload_predict():
    url = "<<Amazon API Gateway's API url link>>"
    if(request.method == "POST"):
        image_file = request.files["image"]
        if(image_file ):
            #image_location = f"{UPLOAD_FOLDER}\{image_file.filename}"
            image_location = f"static\{image_file.filename}" 
            image_file.save(image_location)
            output = predict(image_location,url )
            return 	render_template("index.html",prediction = output, image_loc = image_file.filename)
    return render_template("index.html",prediction = "catt", image_loc = None)

if __name__ =="__main__":
    app.run(port = 12000, debug = True)