from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import QFileDialog
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import login
import home
import add
import err_add
import uhome
import error_log
import err_img

import MySQLdb

import numpy as np
import cv2
import os
import time

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import matplotlib.pyplot as plt


from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras import backend as K

import operator


fname=""
pred=""

class Login(QtGui.QMainWindow, login.Ui_UserLogin):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.log)
        self.pushButton_2.clicked.connect(self.can)
        self.pushButton_3.clicked.connect(self.addNew1)
        
    def log(self):
        i=0
        db = MySQLdb.connect("localhost","root","root","socio")
        cursor = db.cursor()
        a=self.lineEdit.text()
        b=self.lineEdit_2.text()
        sql = "SELECT * FROM users WHERE username='%s' and password='%s'" % (a,b)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                i=i+1
        except Exception as e:
           print e
        
        if i>0:
            if a=="admin":
                print "admin login success"
                self.hide()
                self.home=home()
                self.home.show()
            else:
                print "user login success"
                self.hide()
                self.uhome=uhome()
                self.uhome.show()
            
        else:
            print "login failed"
            self.errlog=errlog()
            self.errlog.show()
                    
        db.close()

    def addNew1(self):
        self.addNew=addNew()
        self.addNew.show()
        
    def can(self):
        sys.exit()
        
   
class addNew(QtGui.QMainWindow, add.Ui_AdNewUser):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save1)
        self.pushButton_3.clicked.connect(self.can2)
        

    def can2(self):
        self.hide()
        #sys.exit()
        
    def save1(self):
        db1 = MySQLdb.connect("localhost","root","root","socio")
        cursor1 = db1.cursor()
        name=self.lineEdit.text()
        email=self.lineEdit_2.text()
        contact=self.lineEdit_3.text()
        uname=self.lineEdit_4.text()
        pwd=self.lineEdit_5.text()
        sql = "INSERT INTO users(name, email, contact, username, password) VALUES ('%s', '%s', '%s', '%s', '%s' )" % (name,email,contact,uname,pwd)
        try:
                cursor1.execute(sql)
                self.hide()
                db1.commit()
        except Exception as e:
                print e
                db1.rollback()
                self.erradd=erradd()
                self.erradd.show()
            

        db1.close()
        

class home(QtGui.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.selimg)
        self.pushButton_2.clicked.connect(self.seldir)
        self.pushButton_3.clicked.connect(self.cnn)
        self.pushButton_5.clicked.connect(self.ex)
        self.pushButton_6.clicked.connect(self.preproc)
        self.pushButton_7.clicked.connect(self.pred)

    def selimg(self):
        global fname
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print fname
        label = QLabel(self.label_5)
        pixmap = QPixmap(fname)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

    
    def seldir(self):
        self.QFileDialog = QtGui.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print folder
        

    def preproc(self):
        global fname
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        else:
            filename = fname
            print "file for processing",filename
            image =cv2.imread(str(filename))
            #print type(image)
            cv2.imshow("Original Image", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("1 - Grayscale Conversion", gray)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("2 - Bilateral Filter", gray)
            edged = cv2.Canny(gray, 27, 40)
            cv2.imshow("4 - Canny Edges", edged)

    def cnn(self):
        #init the model
        model= Sequential()

        #add conv layers and pooling layers 
        model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(32,3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.5)) #to reduce overfitting

        model.add(Flatten())

        #Now two hidden(dense) layers:
        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        model.add(Dropout(0.5))#again for regularization

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))


        model.add(Dropout(0.5))#last one lol

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        #output layer
        model.add(Dense(output_dim = 4, activation = 'sigmoid'))


        #Now copile it
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        #Now generate training and test sets from folders

        train_datagen=ImageDataGenerator(
                                           rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.,
                                           horizontal_flip = False
                                         )

        test_datagen=ImageDataGenerator(rescale=1./255)

        training_set=train_datagen.flow_from_directory("Datasets/training_set",
                                                       target_size = (400,400),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')

        test_set=test_datagen.flow_from_directory("Datasets/test_set",
                                                       target_size = (400,400),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')






        #finally, start training
        hiss=model.fit_generator(training_set,
                                 samples_per_epoch = 890,
                                 nb_epoch = 10,
                                 validation_data = test_set,
                                 nb_val_samples = 320)


        plt.figure(figsize=(15,7))

        plt.subplot(1,2,1)
        plt.plot(hiss.history['acc'], label='train')
        plt.plot(hiss.history['val_acc'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(hiss.history['loss'], label='train')
        plt.plot(hiss.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        #saving the weights
        model.summary()

        model.save_weights("weights.hdf5",overwrite=True)

        #saving the model itself in json format:
        model_json = model.to_json()
        with open("model.json", "w") as model_file:
            model_file.write(model_json)
        print("Model has been saved.")

    def pred(self):
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
            global fname
            global pred
            classes=['agriculture', 'building', 'road', 'water']
            print fname
            frame=cv2.imread(str(fname))
            frame = cv2.resize(frame, (400,400))
            frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
            frame=frame.reshape((1,)+frame.shape)
            frame=frame.reshape(frame.shape+(1,))
            test_datagen = ImageDataGenerator(rescale=1./255)
            m=test_datagen.flow(frame,batch_size=1)
            y_pred=model.predict_generator(m,1)
            histarray2={'agriculture': y_pred[0][0], 'building': y_pred[0][1], 'road': y_pred[0][2], 'water': y_pred[0][3]}
            update(histarray2)
            print histarray2
            print(classes[list(y_pred[0]).index(y_pred[0].max())])
            print(classes[list(y_pred[0]).index(y_pred[0].max())])
            pred= classes[list(y_pred[0]).index(y_pred[0].max())]
            print "pred=",pred
            
             
        model=load_model()
        realtime()
        img = cv2.imread(str(fname), 1)
        color = ('b','g','r')
        qtdBlue = 0
        qtdGreen = 0
        qtdRed = 0
        totalPixels = 0

        for channel,col in enumerate(color):
            histr = cv2.calcHist([img],[channel],None,[256],[1,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            totalPixels+=sum(histr)
            #print histr
            if channel==0:
                qtdBlue = sum(histr)
            elif channel==1:
                qtdGreen = sum(histr)
            elif channel==2:
                qtdRed = sum(histr)

        qtdBlue = (qtdBlue/totalPixels)*100
        qtdGreen = (qtdGreen/totalPixels)*100
        qtdRed = (qtdRed/totalPixels)*100

        qtdBlue = filter(operator.isNumberType, qtdBlue)
        qtdGreen = filter(operator.isNumberType, qtdGreen)
        qtdRed = filter(operator.isNumberType, qtdRed)
        
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges=cv2.Canny(gray,100,200)
        building=0
        road=0
        agri=0
        water=0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        i=0
        cnts,heir= cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for c in cnts: 	
            peri = cv2.arcLength(c, True) 	
            approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)	
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)	
            i=i+1;
            newImage=img[y:y+h,x:x+w]
            if len(c)<10:
                building+=1
            if len(c)>60 and len(c)<100:
                water+=1
            if len(c)>100 and len(c)<500:
                agri+=1
            if len(c)>500:
                road+=1
                water+=1
            
        print "building=",building
        print "agri=",agri
        print "water=",water
        print "road=",road

        print "building resources %f "%(float(building)*100/i)
        print "agri resources %f "%(float(agri)*100/i)
        print "water resources %f "%(float(water)*100/i)
        print "road resources %f "%(float(road)*100/i)
        br=float(building)*100/i
        rr=float(road)*100/i
        ar=float(agri)*100/i
        wr=float(water)*100/i

        global pred
        print "pred=",pred
        if pred=="agriculture":
            ar=float(ar+10)
        if pred=="road":
            rr=float(rr+10)
        if pred=="building":
            br=float(br+10)
        if pred=="water":
            wr=float(wr+10)
            
        self.lineEdit.setText(str(ar))
        self.lineEdit_2.setText(str(rr))
        self.lineEdit_4.setText(str(wr))
        self.lineEdit_3.setText(str(br))
        if br>10 and ar>10 and wr>10:
            self.lineEdit_6.setText("Socio economic condition of area is very good")
            
        if br>10 and ar>1 and wr>0:
            self.lineEdit_6.setText("Socio economic condition of area is fairly good")
        else:
            self.lineEdit_6.setText("Socio economic condition of area is bad")
       
               
        plt.title("Color Heatmap Of Image")
        plt.show()
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
            
    def ex(self):
        sys.exit()
        
class uhome(QtGui.QMainWindow, uhome.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.uselimg)
        self.pushButton_5.clicked.connect(self.uex)
        self.pushButton_6.clicked.connect(self.upreproc)
        self.pushButton_7.clicked.connect(self.upred)
  

    def uselimg(self):
        global fname
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print fname
        label = QLabel(self.label_5)
        pixmap = QPixmap(fname)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

        

    def upreproc(self):
        global fname
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        else:
            filename = fname
            print "file for processing",filename
            image =cv2.imread(str(filename))
            #print type(image)
            cv2.imshow("Original Image", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("1 - Grayscale Conversion", gray)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("2 - Bilateral Filter", gray)
            edged = cv2.Canny(gray, 27, 40)
            cv2.imshow("4 - Canny Edges", edged)

    def upred(self):
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
            global fname
            global pred
            classes=['agriculture', 'building', 'road', 'water']
            frame=cv2.imread(str(fname))
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
            pred= str(classes[list(y_pred[0]).index(y_pred[0].max())])
            print "pred=",pred
            

        model=load_model()
        realtime()
        img = cv2.imread(str(fname), 1)
        color = ('b','g','r')
        qtdBlue = 0
        qtdGreen = 0
        qtdRed = 0
        totalPixels = 0

        for channel,col in enumerate(color):
            histr = cv2.calcHist([img],[channel],None,[256],[1,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            totalPixels+=sum(histr)
            #print histr
            if channel==0:
                qtdBlue = sum(histr)
            elif channel==1:
                qtdGreen = sum(histr)
            elif channel==2:
                qtdRed = sum(histr)

        qtdBlue = (qtdBlue/totalPixels)*100
        qtdGreen = (qtdGreen/totalPixels)*100
        qtdRed = (qtdRed/totalPixels)*100

        qtdBlue = filter(operator.isNumberType, qtdBlue)
        qtdGreen = filter(operator.isNumberType, qtdGreen)
        qtdRed = filter(operator.isNumberType, qtdRed)
        
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges=cv2.Canny(gray,100,200)
        building=0
        road=0
        agri=0
        water=0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        i=0
        cnts,heir= cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for c in cnts: 	
            peri = cv2.arcLength(c, True) 	
            approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)	
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)	
            i=i+1;
            newImage=img[y:y+h,x:x+w]
            if len(c)<10:
                building+=1
            if len(c)>60 and len(c)<100:
                water+=1
            if len(c)>100 and len(c)<500:
                agri+=1
            if len(c)>500:
                road+=1
                water+=1
            
        print "building=",building
        print "agri=",agri
        print "water=",water
        print "road=",road

        print "building resources %f "%(float(building)*100/i)
        print "agri resources %f "%(float(agri)*100/i)
        print "water resources %f "%(float(water)*100/i)
        print "road resources %f "%(float(road)*100/i)
        br=float(building)*100/i
        rr=float(road)*100/i
        ar=float(agri)*100/i
        wr=float(water)*100/i
        global pred
        print "pred=",pred
        if pred=="agriculture":
            ar=float(ar+10)
        if pred=="road":
            rr=float(rr+10)
        if pred=="building":
            br=float(br+10)
        if pred=="water":
            wr=float(wr+10)
            
        self.lineEdit.setText(str(ar))
        self.lineEdit_2.setText(str(rr))
        self.lineEdit_4.setText(str(wr))
        self.lineEdit_3.setText(str(br))
        if br>10 and ar>10 and wr>10:
            self.lineEdit_6.setText("Socio economic condition of area is very good")
            
        if br>10 and ar>1 and wr>0:
            self.lineEdit_6.setText("Socio economic condition of area is fairly good")
        else:
            self.lineEdit_6.setText("Socio economic condition of area is bad")
       
               
        plt.title("Color Heatmap Of Image")
        plt.show()
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
            
    def uex(self):
        sys.exit()
        
class errlog(QtGui.QMainWindow, error_log.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class errimg(QtGui.QMainWindow, err_img.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class erradd(QtGui.QMainWindow, err_add.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()


def main():
    app = QtGui.QApplication(sys.argv)  
    form = Login()                 
    form.show()                         
    app.exec_()                         


if __name__ == '__main__':              
    main()                             
