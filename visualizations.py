from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

filepath = "model_2"

with open(filepath + '.json', 'rb') as f:
        model = model_from_json(f.read())
        model.load_weights(filepath + '.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir='TB')
