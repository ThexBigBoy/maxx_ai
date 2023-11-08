import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Set up data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up train and test data generators
# path_to_train_directory = 
# PATH = os.path.join('_nsfw_train')
# path_to_train_directory = os.path.join(PATH, 'train')
# print(path_to_train_directory)
# path_to_test_directory = os.path.join(PATH, 'testing')
train_generator = train_datagen.flow_from_directory('D:/School Work/maxx_ai/_nsfw_train/train',
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory('D:/School Work/maxx_ai/_nsfw_train/validation',
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='binary')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
        #   steps_per_epoch=total_train_samples // batch_size,
          epochs=40,
          validation_data=test_generator,
        #   validation_steps=total_test_samples // batch_size
          )

loss, accuracy = model.evaluate(test_generator)
print('Test loss:', loss)
print('Test accuracy:', accuracy)