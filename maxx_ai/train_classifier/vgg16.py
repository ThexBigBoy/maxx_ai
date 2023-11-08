# !wget --no-check-certifisafee \
#     https://storage.googleapis.com/mledu-datasets/safe_and_unsafe_filtered.zip \
#     -O /tmp/safe_and_unsafe_filtered.zip

import os
import zipfile
from matplotlib import pyplot as plt
import tensorflow
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import layers
# import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# local_zip = '/tmp/safe_and_unsafe_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp')
# zip_ref.close()

base_dir = 'D:/School Work/maxx_ai/_nsfw_train'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training safe pictures
train_safe_dir = os.path.join(train_dir, 'safe')

# Directory with our training unsafe pictures
train_unsafe_dir = os.path.join(train_dir, 'unsafe')

# Directory with our training sexy pictures
train_sexy_dir = os.path.join(train_dir, 'sexy')

# Directory with our validation safe pictures
validation_safe_dir = os.path.join(validation_dir, 'safe')

# Directory with our validation unsafe pictures
validation_unsafe_dir = os.path.join(validation_dir, 'unsafe')

# Directory with our validation sexy pictures
validation_sexy_dir = os.path.join(validation_dir, 'sexy')


# Set up matplotlib fig, and size it to fit 4x4 pics
import matplotlib.image as mpimg
nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index = 100
train_safe_fnames = os.listdir( train_safe_dir )
train_unsafe_fnames = os.listdir( train_unsafe_dir )
train_sexy_fnames = os.listdir( train_sexy_dir )


next_safe_pix = [os.path.join(train_safe_dir, fname) 
                for fname in train_safe_fnames[ pic_index-8:pic_index] 
               ]

next_unsafe_pix = [os.path.join(train_unsafe_dir, fname) 
                for fname in train_unsafe_fnames[ pic_index-8:pic_index]
               ]

next_sexy_pix = [os.path.join(train_sexy_dir, fname) 
                for fname in train_sexy_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_safe_pix+next_unsafe_pix+next_sexy_pix, 1):
  # Set up subplot; subplot indices start at 1
  try:
    sp = plt.subplot(nrows, ncols, i)
    sp.axis('Off') # Don't show axes (or gridlines)
  except ValueError:
        break

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classifisafeion output
x = layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(base_model.input, x)

model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

# vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 30)

vgghist = model.fit(train_generator, validation_data = validation_generator, epochs = 50)
