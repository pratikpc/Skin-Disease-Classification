from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.preprocessing import image
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to disable tensorflow avx warning



from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
global graph
graph = tf.get_default_graph()



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)





@app.route("/")
def index():
    return "Flask App! Visit <a href='http://192.168.33.20/hello/Jackson/'>http://192.168.33.20/hello/Jackson/</a> " 

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	#model = ResNet50(weights="imagenet")
	#model = load_model('model.h5')
	
	#model = tf.keras.models.load_model('model.h5', custom_objects={'categorical_accuracy':categorical_accuracy, 'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})
	
	
	
# Model reconstruction from JSON file
	with open('model_2.json', 'rb') as f:
	    model = model_from_json(f.read())
        

	# Load weights into the new model
	model.load_weights('model_2.h5')
	
def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

def predict_disease(image):

	with graph.as_default():
		preds = model.predict(image) 
	max_pred = preds.argmax(axis=1)[0]
	return max_pred

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))
			preds = predict_disease(image)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			'''
			with graph.as_default():
				preds = model.predict(image)
			#results = imagenet_utils.decode_predictions(preds)
			'''
			class_weights=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
			data["predictions"] = "You have contacted " + class_weights[preds]
			'''
			# loop over the results and add them to the list of
			# returned predictions
			for (prob,label) in zip(preds[0], class_weights):
			#for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)
			'''
			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
