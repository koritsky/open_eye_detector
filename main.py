import tensorflow as tf
import numpy as np

from utils import predict

model_path = 'model'
model = tf.keras.models.load_model(model_path)
model.compile()

def openEyeChech(inplm):
    """Predicts the state of the eye
    Arguments:
        inplm (str) -- path to the image

    Returns:
        (int) -- 0 if eye is closed, 1 if open
    """

    img = tf.keras.preprocessing.image.load_img(inplm, color_mode='grayscale') # upload image
    img = tf.keras.preprocessing.image.img_to_array(img) #convert to array
    img = np.array([img]) #make a batch
    result = predict(model, img).numpy()[0][0] #get prediction

    return result


