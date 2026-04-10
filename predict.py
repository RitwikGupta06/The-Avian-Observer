import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("bird_model.h5")

# Load class names
with open("classes.txt") as f:
    class_names = [line.strip() for line in f]

# Load image
img_path = "test.jpg"   # change this
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
index = np.argmax(pred)
confidence = np.max(pred)

print("Bird:", class_names[index])
print("Confidence:", confidence)