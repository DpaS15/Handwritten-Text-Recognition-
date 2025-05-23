from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
from PIL import Image
import tensorflow as tf

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

# ✅ Load the model with custom layer
model = load_model("handwriting_model.h5", compile=False, custom_objects={"CTCLayer": CTCLayer})

# Characters used in training (replace with your own if needed)
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((128, 32))  # NOTE: your model expects (128, 32, 1)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Add batch & channel dims
    return img_array

def decode_prediction(prediction):
    pred = np.argmax(prediction, axis=-1)
    result = ""

    # Check if it's a scalar (single prediction) or a sequence
    if np.ndim(pred) == 1:
        pred = [pred]  # wrap in list to make it iterable

    for idx in pred[0]:
        if isinstance(idx, (np.integer, int)) and idx < len(characters):
            result += characters[idx]
    return result

# Final callable function to recognize text
def recognize_text(image_path):
    processed_image = preprocess_image(image_path)
    dummy_labels = np.zeros((1, 16))  # dummy labels needed for model input shape
    prediction = model.predict([processed_image, dummy_labels])
    return decode_prediction(prediction)
