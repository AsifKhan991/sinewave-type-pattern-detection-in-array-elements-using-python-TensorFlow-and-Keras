import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("NN.model")

x_test=[2,3,4,3,2,1,0,-1,-2,-3,-2,-1,-0,-0,-0,-2,-4,-8,4,2,0,-2,-4,-8,-4,-2,0,0,0]
prediction=model.predict([x_test])
print(prediction)
print(np.argmax(prediction))

