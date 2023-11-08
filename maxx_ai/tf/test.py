import tensorflow as tf
import numpy as np
import PIL.Image as Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class_names = ['safe', 'unsafe']


model = tf.keras.models.load_model('nsfw_model.h5')

image = Image.open("1666197121_1-titis-org-p-jennie-leak-nudes-erotika-1.jpg")
image = image.resize((224, 224))  # Resize to the input size expected by the model
image = np.array(image) / 255.0  # Normalize pixel values

predictions = model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(predictions)


predictions = model.predict(np.expand_dims(image, axis=0))
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

for i in score:
    # print(i)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(i)], 100 * np.max(i))
    )
    # print(class_names[np.argmax(i)])

print("Predicted class:", predicted_class)