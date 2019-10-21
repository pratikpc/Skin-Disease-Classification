# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
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
import sys
from sklearn.metrics import  confusion_matrix
import scikitplot as skplt
from keras.models import model_from_json
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# dimensions of our images
img_width, img_height = 224,224



class_weights=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
metrics = ['precision', 'recall', 'f1score']
count = 0; total = 0
ypred = []; ytrue= []
image_dir = "data"

def predict_disease(images, model):
    
        img = os.path.join(image_dir, images)
        from keras.preprocessing import image

        img = image.load_img(img, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        image = imagenet_utils.preprocess_input(x)

        preds = model.predict(image) 
        max_pred = preds.argmax(axis=1)[0]
        return max_pred

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return cm

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

def calc_FPR(true_n, false_p):
    if true_n + false_p == 0:
        return 0.0
    else:
        return false_p / (true_n + false_p)


def precision_recall_fscore(cm):
    precision = dict(); precision[" "] = "precision"
    recall = dict(); recall[" "] = "recall"
    fscore = dict(); fscore[" "] = "fscore"
    FPR = dict(); FPR[" "] = "FPR"
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
        FPR[weight] = calc_FPR(true_n, false_p)
    precision_total = calc_precision(true_p_total, false_p_total)
    recall_total = calc_precision(true_p_total, false_n_total)
    fscore_total = calc_precision(precision_total, recall_total)
    FPR_total = calc_FPR(true_n_total, false_p_total)
    return precision, recall, fscore, FPR,  precision_total, recall_total, fscore_total, FPR_total
        
if __name__ == '__main__':
    filepath = sys.argv[1]
    with open(filepath + '.json', 'rb') as f:
        model = tf.keras.models.model_from_json(f.read())
        model.load_weights(filepath + '.h5')
    with open("test.csv" , "r") as csv_file:
        data = csv.reader(csv_file)
        for row in data:
                if total == 0:
                        total += 1
                        continue
                preds = predict_disease(row[1] + ".jpg", model)
                weight = class_weights[preds]
                ypred.append(class_weights[preds])
                ytrue.append(row[2])
    
    np.set_printoptions(precision=2)

    cm = plot_confusion_matrix(ytrue, ypred, classes=class_weights,
                      title='Confusion matrix, without normalization')

    cm = np.matrix.transpose(cm)
    plt.show()
    precision, recall, fscore, FPR,  precision_total, recall_total, fscore_total, FPR_total = precision_recall_fscore(cm)
    print("Total precision = ", precision_total)
    print("Total recall = ", recall_total)
    print("Total fscore = ", fscore_total)

    '''
    class_weights = list(" ") + class_weights
    with open(filepath + ".csv" , "w", newline = "\n", ) as csvfile:
        writer = csv.DictWriter(csvfile, class_weights)
        writer.writeheader()
        writer.writerow(precision)
        writer.writerow(recall)
        writer.writerow(fscore)
    '''







    