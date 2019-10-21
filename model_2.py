# The model for the skin cancer classifier

# Import the libraries
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to disable tensorflow avx warning


# Check if GPU is available
K.tensorflow_backend._get_available_gpus()

# The paths for the training and validation images
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

# Declare a few useful values
num_train_samples = 9013
num_val_samples = 1002
train_batch_size = 10
val_batch_size = 10
image_size = 224

# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Set up generators
train_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size)

valid_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size)

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    shuffle=False)

print(valid_batches.class_indices)
filepath = "model_2"

with open(filepath + '.json', 'rb') as f:
        model = model_from_json(f.read())
        model.load_weights(filepath + '.h5')

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Compile the model
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_5_accuracy])

val_loss, val_cat_acc, val_top_2_acc, val_top_5_acc = \
model.evaluate_generator(test_batches, steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_5_acc)

# Create a confusion matrix of the test images
test_labels = test_batches.classes

# Make predictions
predictions = model.predict_generator(test_batches, steps=val_steps, verbose=1)

# Declare a function for plotting the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def calc_metrics(cm, index):
    true_p =0; true_n = 0; false_p = 0; false_n = 0
    true_p = float(cm[index][index])
    false_p = float(np.sum(cm, axis = 1)[index] - true_p)
    false_n = float(np.sum(cm, axis =0)[index] - true_p)
    true_n = float(np.sum(cm) - false_p - false_n - true_p)

    return true_p, true_n, false_p, false_n

def calc_precision(true_p, false_p):
    if true_p + false_p == 0:
        return 0.0
    else:
        return true_p / (true_p + false_p)

def calc_recall(true_p, false_n):
    if true_p + false_n == 0:
        return 0.0
    else:
        return true_p / (true_p + false_n)

def calc_fscore(precision, recall):
    if precision + recall == 0:
        return 0.0
    else:
        return (2 * precision * recall)/ (precision + recall)

def calc_accuracy(true_p, true_n, false_p,false_n):
    if true_p + true_n + false_p + false_n == 0:
        return 0.0
    else:
        return (true_p + true_n)/ ( true_p + true_n + false_p + false_n)

def precision_recall_fscore(cm, class_weights):
    precision = dict(); precision[" "] = "precision"
    recall = dict(); recall[" "] = "recall"
    fscore = dict(); fscore[" "] = "fscore"
    accuracy = dict(); accuracy[" "] = "accuracy"
    true_p_total = 0; true_n_total = 0; false_p_total = 0; false_n_total = 0
    for weight in class_weights:
        true_p, true_n, false_p, false_n = calc_metrics(cm, class_weights.index(weight))
        true_p_total += true_p
        true_n_total += true_n
        false_p_total += false_p
        false_n_total += true_n
        precision[weight] = calc_precision(true_p, false_p)
        recall[weight] = calc_recall(true_p, false_n)
        fscore[weight] = calc_fscore(precision[weight],recall[weight])
        accuracy[weight] = calc_accuracy(true_p, true_n, false_p,false_n)
    precision_total = calc_precision(true_p_total, false_p_total)
    recall_total = calc_recall(true_p_total, false_n_total)
    fscore_total = calc_fscore(precision_total, recall_total)
    return precision, recall, fscore, precision_total, recall_total, fscore_total, accuracy

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']
plot_confusion_matrix(cm, cm_plot_labels)
plt.show()

cm = np.matrix.transpose(cm)
precision, recall, fscore, precision_total, recall_total, fscore_total, accuracy = precision_recall_fscore(cm, cm_plot_labels)
print("Micro Averaging precision = ", precision_total)
print("Micro Averaging recall = ", recall_total)
print("Micro Averaging fscore = ", fscore_total)



macro_averaging_precision = 0
macro_averaging_recall = 0
accuracy_total = 0
for weight in cm_plot_labels:
    macro_averaging_precision += precision[weight]
    macro_averaging_recall += recall[weight]
    accuracy_total += accuracy[weight]
macro_averaging_precision = macro_averaging_precision / 7
macro_averaging_recall = macro_averaging_recall / 7
accuracy_total = accuracy_total / 7
print("Accuracy is ", accuracy_total)
macro_averaging_fscore = calc_fscore(macro_averaging_precision, macro_averaging_recall)
metric_list = ['micro', 'macro']
precision['micro'], precision['macro'] = precision_total, macro_averaging_precision
recall['micro'], recall['macro'] = recall_total, macro_averaging_recall
fscore['micro'], fscore['macro'] = fscore_total, macro_averaging_fscore

class_weights = list(" ") + cm_plot_labels + metric_list
with open(filepath + ".csv" , "w", newline = "\n", ) as csvfile:
    writer = csv.DictWriter(csvfile, class_weights)
    writer.writeheader()
    writer.writerow(precision)
    writer.writerow(recall)
    writer.writerow(fscore)



