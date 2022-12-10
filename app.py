# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 01:31:48 2022

@author: Hp
"""

import cv2
import streamlit as st 
from PIL import Image
from werkzeug.utils import secure_filename
import pickle 
from sklearn.preprocessing import StandardScaler
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import pandas as pd
from matplotlib.image import imread
from skimage.io import imread_collection
from PIL import Image
import seaborn as sns
from sklearn import decomposition, preprocessing, svm
import sklearn.metrics as metrics 
from time import sleep 
import os
sns.set()


st.set_option('deprecation.showfileUploaderEncoding', False)
model = pickle.load(open('Hackathon.pkl', 'rb'))  


html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Machine Learning/Artificial Intelligence</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Alzheimer Detection System
         """
         )
file = st.file_uploader("Please upload an image", type=("jpg", "png"))

def predict(image):
    prediction = model.predict(image)
    if prediction[0] == 1:
      val = "Person has Alzimer."
    else:
      val = "Person does not has Alzimer"
    return val

def resizer(image):
    width = 256
    height = 256
    new_size = (width,height)
    img = Image.open(image)
    img = img.resize(new_size)
    array_temp = np.array(img)
    shape_new = width*height
    img_wide = array_temp.reshape(1, shape_new)
    output = predict(img_wide)
    return output


if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image=np.array(image)
  st.image(image,caption='Uploaded Image.', use_column_width=True)

if st.button("Predict Expression"):
  result=resizer(file)
  st.success('Model has predicted that {}'.format(result))
if st.button("About"):
  st.header("Team Infinity")
  st.subheader("Students, Department of Computer Engineering")
  
html_temp = """
   <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"></script>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
   </head>
   <body>
  
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Project</p></center> 
   </div>
   </div>
   </div>
   <div class="fixed-bottom ">
      <div class="dark bg-dark " style="min-height: 40px;">
         <marquee style="color:#fff; margin-top: 7px;">
            <h9>Designed & Developed by Team Infinity, Students of Poornima Institute of Engineering and Technology</h9>
         </marquee>
      </div>
   </div>
   </body>
   """
st.markdown(html_temp,unsafe_allow_html=True)
