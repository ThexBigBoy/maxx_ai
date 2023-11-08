import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory('D:/School Work/maxx_ai/_nsfw_train/train',
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary')

test_dataset = test_datagen.flow_from_directory('D:/School Work/maxx_ai/_nsfw_train/validation',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='binary')

# Load a pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in model.layers:
    layer.trainable = False

# Add custom classification layers
x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset,
          epochs=10,
          validation_data=test_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(test_dataset)