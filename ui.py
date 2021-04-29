from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk,Image
import os 
import cv2
import time
import random
import argparse
import numpy as np 
import string
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense, Layer, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#External files
from utils import *

# Datapath of Datasets
datapath= "Datasets/samples/samples"
symbols = string.ascii_lowercase + '0123456789'
len_symbols = len(string.ascii_lowercase + "0123456789")

global img 

def preprocessing_1(path):
    
    console.config(state=NORMAL)
    console.insert(END,"Processing Image...\n")
    console.config(state=DISABLED)

    img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
    targets = path.split('.')[0]
    targ = 0
    X = np.zeros((1 , 50 , 200 ,1 ))  
    y = np.zeros((1,100, 36 ))
    


    print(targets)
    print(len(targets))
    img = img/255.0
    img = np.reshape(img , (50,200,1))
    targ = np.zeros((100,36))
    for l , char in enumerate(targets):
        idx = symbols.find(char)
        targ[l , idx] = 1
    X[0] = img
    y[0,: ,:] = targ
    console.config(state=NORMAL)
    console.insert(END,"Processing finished...\n")
    console.config(state=DISABLED)
    return X,y

def predictions(image):
    image = np.reshape(image , (50,200))
    result = model.predict(np.reshape(image , (1,50,200,1)))
    result = np.reshape(result ,(5,36))
    indexes =[]
    for i in result:
        indexes.append(np.argmax(i))
        
    label=''
    for i in indexes:
        label += symbols[i]
        
    plt.imshow(image)
    plt.title(label)

    rectxt.config(text = label)
    


def preprocessing(path):
    
    n_samples= len(os.listdir(path))
	# variables for data and labels 
    X = np.zeros((n_samples , 50 , 200 ,1 ))  # (samples , height , width , channel)
    y = np.zeros((n_samples,5, 36 ))       #(samples , captcha characters , ascii char + numbers)

    for i , image in enumerate(os.listdir(path)):
        img = cv2.imread(os.path.join(path, image) , cv2.IMREAD_GRAYSCALE)

        targets = image.split('.')[0]

        if len(targets)<6:
            img = img/255.0
            img = np.reshape(img , (50,200,1))

            #find the char and one hot encode it to the target
            targ = np.zeros((5,36))

            for l , char in enumerate(targets):

                idx = symbols.find(char)
                targ[l , idx] = 1
            X[i] = img
            y[i,: ,:] = targ
            
    console.config(state=NORMAL)
    console.insert(END,"Preprocessing Finished")
    console.insert(END,"\n")
    return X,y


     

def myModel():
   
    inputs = Input(shape=(50,200,1) , name='image')
    x= Conv2D(16, (3,3),padding='same',activation='relu')(inputs)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x = BatchNormalization()(x)
    out_flat= Flatten()(x)
    
    #char-1
    dense_1 = Dense(64 , activation='relu')(out_flat)
    dropout_1= Dropout(0.5)(dense_1)
    out_1 = Dense(len_symbols , activation='sigmoid' , name='char_1')(dropout_1)
    
    #char-2
    dense_2 = Dense(64 , activation='relu')(out_flat)
    dropout_2= Dropout(0.5)(dense_2)
    out_2 = Dense(len_symbols , activation='sigmoid' , name='char_2')(dropout_2)
    
    #char-3
    dense_3 = Dense(64 , activation='relu')(out_flat)
    dropout_3= Dropout(0.5)(dense_3)
    out_3 = Dense(len_symbols , activation='sigmoid' , name='char_3')(dropout_3)
    
    #char-4
    dense_4 = Dense(64 , activation='relu')(out_flat)
    dropout_4= Dropout(0.5)(dense_4)
    out_4 = Dense(len_symbols , activation='sigmoid' , name='char_4')(dropout_4)
    
    #char-5
    dense_5 = Dense(64 , activation='relu')(out_flat)
    dropout_5= Dropout(0.5)(dense_5)
    out_5 = Dense(len_symbols , activation='sigmoid' , name='char_5')(dropout_5)
    
    model_out = Model(inputs=inputs , outputs=[out_1 , out_2 , out_3 , out_4 , out_5])
    
    return model_out


def train():
    
    global model
    model = myModel()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])

    X, y = preprocessing(datapath)
    trainX , testX , trainY , testY = train_test_split(X, y , test_size=0.2 , random_state=42)
    
    labels = {'char_1': trainY[:,0,:], 
         'char_2': trainY[:,1,:],
         'char_3': trainY[:,2,:],
         'char_4': trainY[:,3,:],
         'char_5': trainY[:,4,:]}

    test_labels = {'char_1': testY[:,0,:], 
            'char_2': testY[:,1,:],
            'char_3': testY[:,2,:],
            'char_4': testY[:,3,:],
            'char_5': testY[:,4,:]}

    history = model.fit(trainX ,labels , epochs=30 , batch_size=64 , validation_split=0.2)

    console.config(state=NORMAL)
    console.insert(END,"Caluclating score...\n")
    console.config(state=DISABLED)

    score =model.evaluate(testX , test_labels , batch_size=32)

    console.config(state=NORMAL)
    console.insert(END,"\n")
    console.insert(END,"The score of the model is:\n")

    console.insert(END, score)
    console.insert(END,"\n")
    console.config(state=DISABLED)

    console.config(state=NORMAL)
    console.insert(END,"[INFO]: Training Completed")
    console.insert(END,"\n")
    



def brwse():
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("PNG files",
                                                        "*.png*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
    filepath.delete("1.0","end")
    filepath.insert(END, filename)



def solve():
    print(filename)
    # X, y = preprocessing_1(filename)
    # predictions(X[0])


base = Tk()
base.title("CAPTCHA SOLVER")

base.geometry('500x650')
base.resizable(width=False, height=False)
menubar = Menu(base)
filemenu = Menu(menubar, tearoff = 0)
menubar.add_cascade(label = "About", menu = filemenu)
editmenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label = "Exit", command = base.quit)
helpmenu = Menu(menubar, tearoff=0)
base.config(menu = menubar)

P1 = Label(base, text = "CAPTCHA TEXT RECOGNIZER", font=("Arial", 18, "bold"), fg="#0000ff")
P1.place(relx = 0.5,rely = 0.03,anchor ='center')

path_text = Text(base, bd=0, bg="White", height="1", width="20", font=("Arial", 12))


b_preprocessing =Button(base, text="Train", height="1", width="15", command=train)
b_preprocessing.place(relx = 0.73,rely = 0.12,anchor ='sw')

imgtxt = Label(base, text = "File path:", font=("Arial", 12))
imgtxt.place(relx = 0.04,rely = 0.18,anchor ='sw')


filepath = Text(base, bd=0, bg="White", height="1", width="35", font=("Arial", 10))
filepath.place(relx = 0.19,rely = 0.18,anchor ='sw')

b_preprocessing =Button(base, text="Browse", height="1", width="15", command=brwse)
b_preprocessing.place(relx = 0.73,rely = 0.18,anchor ='sw')


solve =Button(base, text="Solve Captcha", height="2", width="64", command=solve)
solve.place(relx = 0.5,rely = 0.25,anchor ='center')

smpltxt = Label(base, text = "Original Image:", font=("Arial", 12, "bold"))
smpltxt.place(relx = 0.04,rely = 0.35,anchor ='sw')

canvas = Canvas(base, width = 300, height = 300)      
canvas.place(relx = 0.55,rely = 0.58,anchor ='center')      


smpltxt = Label(base, text = "Recognised Text:", font=("Arial", 12, "bold"))
smpltxt.place(relx = 0.04,rely = 0.54,anchor ='sw')


rectxt = Label(base, text ="", font=("Arial", 20, "bold"))
rectxt.place(relx = 0.04,rely = 0.60,anchor ='sw')

smpltxt = Label(base, text = "Console:", font=("Arial", 12, "bold"))
smpltxt.place(relx = 0.04,rely = 0.68,anchor ='sw')

console = Text(base, bd=0, bg="White", height="8" ,width="65", font=("Consolas",10))
console.place(relx = 0.04,rely = 0.87,anchor ='sw')
console.config(state=DISABLED)

console.config(state=NORMAL)
console.insert(END,"[INFO]: Starting...")
console.insert(END,"\n")


base.mainloop()
