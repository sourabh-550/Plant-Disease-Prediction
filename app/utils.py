import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

IMG_SIZE = 224

class PlantDiseaseModel:
    
    def __init__(self, model_path, class_names_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

        print("Loaded Model Input Shape:", self.model.input_shape)
    
    def preprocess_image(self, img):
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, img):
        processed_img = self.preprocess_image(img)
        predictions = self.model.predict(processed_img)
        
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_label = self.class_names[predicted_index]
        
        return predicted_label, confidence