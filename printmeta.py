from AlphaGo.models.policy import CNNPolicy
from keras.utils.visualize_util import plot

MODEL = '/alphago/SLv1/my_model.json'
WEIGHTS = '/alphago/SLv1/weights.00002.hdf5'
policy = CNNPolicy.load_model(MODEL)



plot(model, to_file='model.png')