import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import load_model
from nudenet.image_utils import load_images

class Classifier:
    nsfw_model = None

    def __init__(self):
        dirname = os.path.dirname(__file__)
        model_path = os.path.join(dirname, "models/nsfw_model.h5")
        self.nsfw_model = load_model(model_path)

    def classify(
        self,
        image_paths=[],
        batch_size=4,
        image_size=(224, 224),
        categories=["safe", "unsafe"],
    ):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )

        if not loaded_image_paths:
            return {}

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
                loaded_image_path = str(i)
            images_preds[loaded_image_path] = {}
            for j, pred in enumerate(preds[i]):
                images_preds[loaded_image_path][pred] = float(probs[i][j])

        return images_preds