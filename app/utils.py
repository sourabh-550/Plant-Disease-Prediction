import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

IMG_SIZE = 224

class PlantDiseaseModel:
    
    def __init__(self, model_path, data_dir):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = sorted(os.listdir(data_dir))
    
    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def predict(self, img_path):
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img)
        
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_label = self.class_names[predicted_index]
        
        return {
            "class": predicted_label,
            "confidence": confidence
        }