# -*- coding: utf-8 -*-


from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import threading
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K



histarray={'agriculture':0, 'building':0, 'road': 0, 'water': 0}


def load_model():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except Exception as e:
        print e
        print("""Model not found. Please train the CNN by running the script """)
        return None
    
    

def update(histarray2):
    global histarray
    histarray=histarray2



def realtime():
       
    classes=['agriculture', 'building', 'road', 'water']
    
    frame=cv2.imread('123.png')
    frame = cv2.resize(frame, (400,400))
    frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
    frame=frame.reshape((1,)+frame.shape)
    frame=frame.reshape(frame.shape+(1,))
    test_datagen = ImageDataGenerator(rescale=1./255)
    m=test_datagen.flow(frame,batch_size=1)
    y_pred=model.predict_generator(m,1)
    histarray2={'agriculture': y_pred[0][0], 'building': y_pred[0][1], 'road': y_pred[0][2], 'water': y_pred[0][3]}
    update(histarray2)
    print(classes[list(y_pred[0]).index(y_pred[0].max())])
     
    

model=load_model()
realtime()

    
