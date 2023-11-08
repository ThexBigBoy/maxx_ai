import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input_shape = (224, 224, 3)

datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values to [0, 1]
                             preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,  # Preprocess images for MobileNetV2
                             validation_split=0.2)  # Split the dataset into training and validation sets

batch_size = 32
train_generator = datagen.flow_from_directory(
    'D:/School Work/maxx_ai/_nsfw_train/train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'D:/School Work/maxx_ai/_nsfw_train/validation',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modify last layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='softmax')(x)

# Create new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data
train_data_dir = 'D:/School Work/maxx_ai/_nsfw_train/train'  # Replace with the path to your training data directory
validation_data_dir = 'D:/School Work/maxx_ai/_nsfw_train/validation'  # Replace with the path to your validation data directory
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the trained model
model.save('trained_model.h5')  # Replace with the desired path and filename for the saved model