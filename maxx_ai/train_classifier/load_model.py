
# # load and evaluate a saved model
# from numpy import loadtxt
# import tensorflow
# from keras.models import load_model
 
# # load model
# model = load_model('rps.h5')
# # summarize model.
# model.summary()
# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Recreate the exact same model, including its weights and the optimizer
import tensorflow as tf


new_model = tf.keras.models.load_model('rps.h5')

# Show the model architecture
new_model.summary()