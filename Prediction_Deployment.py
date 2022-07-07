# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:07:15 2022

@author: DELL
"""

# Import Statements

import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

# load json and create model
json_file = open('C:/Users/DELL/OneDrive/Desktop/Model_6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/DELL/OneDrive/Desktop/Model_6.h5")

# Prediction Label
def Prediction(Pred):
    if (Pred==1):
        st.write("The Patient has Parkinson's Disease")
    elif(Pred==0):
        st.write('Normal Patient')

#st. title('Deep Learning Based Parkinsons Disease Prediction')       
title = '<p style="font-family:sans-serif; color:	#000080; font-size: 28px;">Deep Learning Based Parkinsons Disease Prediction</p>'
st.markdown(title, unsafe_allow_html=True)


#uploaded_files = st.file_uploader("Choose an Image file", accept_multiple_files=True)
#for uploaded_file in uploaded_files:  
     #st.write("\\", uploaded_file.name)
     
# Load Image     
img_up=st.file_uploader('Select an Image file for Display')

if img_up is not None:
    
    imag=Image.open(img_up)
    
    st.image(imag,caption= 'upload image')

# Enter file name
f_name = st.text_input("Enter file name for Prediction: ")
path_img=r"C:\Users\DELL\M.TECH\DEEP LEARNING\prj\P_D_DETECTION\MODEL_WEIGHTS"
img=path_img+f_name

if st.button('Predict'):
    
    i = tf.keras.preprocessing.image.load_img(img, target_size=(256,256))
    i = tf.keras.preprocessing.image.img_to_array(i)/255.0
    i = i.reshape(1, 256,256,3)
    p=np.argmax(loaded_model.predict(i))
    Prediction(p)
    