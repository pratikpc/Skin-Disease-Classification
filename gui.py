# import the necessary packages
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
from keras.models import load_model

from keras.applications import imagenet_utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to disable tensorflow avx warning
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

img_width, img_height = 224,224
from keras.models import model_from_json
with open('model_2.json', 'rb') as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights('model_2.h5')

filepath = "model.h5"
class_weights=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
pred_list = []
max_pred_list = []
image_dir = "data"

def predict_disease(images):
    from keras.preprocessing import image as kimage
    img = kimage.load_img(images, target_size=(img_width, img_height))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img = imagenet_utils.preprocess_input(x)

    preds = model.predict(img) 
    max_pred = preds.argmax(axis=1)[0]
    return max_pred

def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        #image = cv2.imread(path)
        preds = predict_disease(path)
        weight = class_weights[preds]


        image = cv2.imread(path)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="top", padx=10, pady=10)

        # while the second panel will store the edge map
        panelB = Label(text = weight)
        panelB.text = weight
        panelB.pack(side="bottom", padx=10, pady=10)

    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(text = weight)
        panelA.image = image
        panelB.text = weight

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()