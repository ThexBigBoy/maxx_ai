import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from keras.models import load_model
from nudenet.image_utils import load_images
import tensorflow as tf
import PIL.Image as Image


class Classification:
    """
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
        model = Classifier()
        """
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, "models/nsfw_model.h5")
        self.nsfw_model = load_model(model_path)

    def classify(
        self,
        image_paths=[],
        batch_size=4,
        # batch_size=32,
        # image_size=(256, 256),
        image_size=(224, 224),
        categories=["safe", "unsafe"],
        # class_names = ['safe', 'unsafe'],
        
    ):
        # print(image_paths)
        """
        inputs:
            image_paths: list of image paths or can be a string too (for single image)
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )
        if not loaded_image_paths:
            return {}
        

        # model = tf.keras.models.load_model('filterd1_model.h5')

        # image = Image.open("portrait-beautiful-sexy-girl-with-perfect-body_942478-2570 copy.jpg")
        # image = image.resize((180, 180))  # Resize to the input size expected by the model
        # image = np.array(image) / 255.0  # Normalize pixel values

        # predictions = model.predict(np.expand_dims(image, axis=0))
        # predicted_class = np.argmax(predictions)


        # predictions = model.predict(np.expand_dims(image, axis=0))
        # score = tf.nn.softmax(predictions[0])
        # images_preds = {}

        # images_preds = class_names[np.argmax(score)], "Score: ", [100 * np.max(score)]

        

        # return images_preds
        preds = []
        model_preds = []
        while len(loaded_images):
            _model_preds = self.nsfw_model.predict(
                loaded_images[:batch_size]
            )
            model_preds.append(_model_preds)
            preds += np.argsort(_model_preds, axis=1).tolist()
            loaded_images = loaded_images[batch_size:]

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(
                    model_preds[int(i / batch_size)][int(i % batch_size)][pred]
                )
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i
            images_preds = {}
            for _ in range(len(preds[i])):
                images_preds[preds[i][_]] = float(probs[i][_])

        return images_preds