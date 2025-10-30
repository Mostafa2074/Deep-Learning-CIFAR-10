import tensorflow as tf
import numpy as np
from PIL import Image
from utils import Preprocessing

class ModelHandler:
    def __init__(self, model_path="D:\DEPI ASSIGN 14\Assignment\models\my_model.keras"):
        self.model = tf.keras.models.load_model(model_path)

        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def preprocess_image(self, image_path):
        image = Image.open(image_path).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        predictions = self.model.predict(image)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        label = self.class_names[class_index]
        return label, confidence
