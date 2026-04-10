import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = 224
BATCH_SIZE = 32

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
print("Classes:", class_names)

# Normalize
train_data = train_data.map(lambda x, y: (x/255.0, y))

# Load pretrained model
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, epochs=5)

# Save model + class names
model.save("bird_model.h5")

with open("classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")