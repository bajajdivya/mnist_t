from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
import json
import numpy as np
from tensorflow import Graph
from PIL import ImageOps



img_height, img_width = 28,28
with open ('./models/model.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/mnist.h5')


# Create your views here.

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)
"""
 * - reshape didn't work -> lsdajflasd
"""

def predictImage(request):

    print(request.FILES)
    print(request.POST.dict())
    print(request)
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage() 
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    print(testimage)

    img = tf.keras.utils.load_img(
    testimage,
    grayscale=True,
    )
    
    x= tf.keras.utils.img_to_array(img)
    print(x)

    x = x.reshape(1, 28, 28, 1)
 
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
 

    
    predictedLabel = str(np.argmax(predi[0]))
    # predictedLabel = dict[str(np.argmax(predi[0]))]





    context = {'filePathName': filePathName,'predictedLabel': predictedLabel}
    return render(request, 'index.html',context)

