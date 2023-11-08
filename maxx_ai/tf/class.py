from keras.models import load_model

# Load the image classification model
model = load_model('nsfw_model.h5')

# Access the final layer
final_layer = model.layers[-1]

# Check the shape of the final layer's output
num_classes = final_layer.output_shape[-1]

print("Number of classes:", num_classes)