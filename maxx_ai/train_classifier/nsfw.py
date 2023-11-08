import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Set the path to your dataset
train_data_dir = 'D:/School Work/maxx_ai/_nsfw_train/training'
validation_data_dir = 'D:/School Work/maxx_ai/_nsfw_train/validation'

# Set the parameters
img_width, img_height = 224, 224
batch_size = 32
num_epochs = 10
num_classes = 2  # Number of classes in your dataset

# Preprocess and augment the training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Preprocess the validation data
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Use softmax activation for multi-class classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Use categorical_crossentropy for multi-class classification

# Prepare the training and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical class mode for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical class mode for multi-class classification
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('nsfw_model.h5')