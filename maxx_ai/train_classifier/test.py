import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load the pre-trained MobileNet model without the final classification layer
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
model = hub.KerasLayer(model_url, trainable=False)

# Add a new classification layer
num_classes = 3  # Number of output categories (cats and dogs)
classification_layer = tf.keras.Sequential([
    model,
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
classification_layer.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

# Define the paths to your training and validation datasets
train_dataset_path = "D:/School Work/maxx_ai/_nsfw_train/train"
val_dataset_path = "D:/School Work/maxx_ai/_nsfw_train/validation"

# Set up data augmentation and preprocessing
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(train_dataset_path,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='binary')

val_dataset = val_datagen.flow_from_directory(val_dataset_path,
                                              target_size=image_size,
                                              batch_size=batch_size,
                                              class_mode='binary')

# Train the model
history = classification_layer.fit(train_dataset,
                                   epochs=10,
                                   validation_data=val_dataset)

# Evaluate the model on the validation dataset
loss, accuracy = classification_layer.evaluate(val_dataset)

print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")