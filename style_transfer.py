# %%
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import clear_output, display
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Configure Matplotlib
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# Ensure TensorFlow version is 2.x
print(f"TensorFlow Version: {tf.__version__}")

# -------------------
# Parameters
# -------------------

# Supported Models:
# 'Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet50V2', 'ResNet101',
# 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'InceptionV3', 'MobileNet',
# 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201',
# 'EfficientNetB0' to 'EfficientNetV2L'

model_name = 'VGG16'

# Paths to content and style images
content_image_path = '/content/drive/MyDrive/dev/pics/mac2.jpg'  # Replace with your content image path
style_image_path = '/content/drive/MyDrive/dev/pics/duncan_jago_unit.jpg'     # Replace with your style image path

# Adjust parameters
content_weight = 1e4
style_weight = 1e-2
total_variation_weight = 30
epochs = 10
steps_per_epoch = 20
max_dim = 512  # Max dimension of the images

# Ensure the paths are correct
assert os.path.exists(content_image_path), "Content image not found."
assert os.path.exists(style_image_path), "Style image not found."

# %%
# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)
    tensor = tensor.numpy()
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load and preprocess images
def load_img(path_to_img, max_dim=None):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

# %%
# Function to get the model and preprocessing function
MODEL_CONFIGS = {
    'Xception': {
        'builder': tf.keras.applications.Xception,
        'preprocess': tf.keras.applications.xception.preprocess_input,
        'style_layers': ['block1_conv1', 'block2_sepconv1', 'block3_sepconv1'],
        'content_layers': ['block14_sepconv2_act']
    },
    'VGG16': {
        'builder': tf.keras.applications.VGG16,
        'preprocess': tf.keras.applications.vgg16.preprocess_input,
        'style_layers': [
            'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
        ],
        'content_layers': ['block5_conv2']
    },
    'VGG19': {
        'builder': tf.keras.applications.VGG19,
        'preprocess': tf.keras.applications.vgg19.preprocess_input,
        'style_layers': [
            'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
        ],
        'content_layers': ['block5_conv4']
    },
    # Add other models as needed...
}

def get_model(model_name, style_layers, content_layers):
    # Retrieve config for the chosen model
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model not supported. Choose from: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    base_model = config['builder'](include_top=False, weights='imagenet')
    preprocess_input = config['preprocess']

    base_model.trainable = False
    outputs = [base_model.get_layer(name).output for name in style_layers + content_layers]
    model = tf.keras.Model([base_model.input], outputs)
    return model, preprocess_input

num_content_layers = len(MODEL_CONFIGS[model_name]['content_layers'])
num_style_layers = len(MODEL_CONFIGS[model_name]['style_layers'])

# Get the model and preprocessing function
model, preprocess_input = get_model(model_name,
                                     MODEL_CONFIGS[model_name]['style_layers'],
                                     MODEL_CONFIGS[model_name]['content_layers'])

print(f"Using {model_name} model for style transfer.")
print("Style Layers:")
for layer in MODEL_CONFIGS[model_name]['style_layers']:
    print(f" - {layer}")
print("Content Layers:")
for layer in MODEL_CONFIGS[model_name]['content_layers']:
    print(f" - {layer}")

# %%
# Load the images
content_image = load_img(content_image_path, max_dim)
style_image = load_img(style_image_path, max_dim)

# Display the content and style images
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# %%
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Class to extract style and content
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, model, style_layers, content_layers, preprocess_input):
        super(StyleContentModel, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.preprocess_input = preprocess_input
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = self.preprocess_input(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

# Create an instance of the style and content extractor
extractor = StyleContentModel(model,
                              MODEL_CONFIGS[model_name]['style_layers'],
                              MODEL_CONFIGS[model_name]['content_layers'],
                              preprocess_input)

# Extract style and content targets
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# %%
class StyleContentModelRaw(tf.keras.models.Model):
    def __init__(self, model, style_layers, preprocess_input):
        super(StyleContentModelRaw, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.preprocess_input = preprocess_input

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = self.preprocess_input(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs = outputs[:len(self.style_layers)]  # Only style layers
        return {style_name: output for style_name, output in zip(self.style_layers, style_outputs)}

# Create a raw extractor for visualization
extractor_raw = StyleContentModelRaw(model,
                                     MODEL_CONFIGS[model_name]['style_layers'],
                                     preprocess_input)

# Visualize raw style layers
def visualize_raw_style_layers(style_image, extractor_raw):
    # Extract raw style layer outputs
    style_outputs = extractor_raw(style_image)

    for layer_name, style_output in style_outputs.items():
        style_output = tf.squeeze(style_output).numpy()  # Remove batch dimension
        num_filters = style_output.shape[-1]
        height, width = style_output.shape[0], style_output.shape[1]

        # Determine grid size
        cols = 8
        rows = num_filters // cols + int(num_filters % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        fig.suptitle(f'Style Layer: {layer_name}', fontsize=16)

        for i in range(num_filters):
            if i >= rows * cols:
                break
            ax = axes[i // cols, i % cols]
            ax.imshow(style_output[..., i], cmap='viridis')
            ax.axis('off')

        # Turn off unused subplots
        for j in range(i + 1, rows * cols):
            axes[j // cols, j % cols].axis('off')

        plt.show()

# Visualize raw style layers
visualize_raw_style_layers(style_image, extractor_raw)

# %%
# Cell [6]: Optimization Setup

# Initialize the image to be optimized
image = tf.Variable(content_image)

# Define the optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Function to compute style and content loss
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

# Function to perform a training step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# %%
class StyleContentModelRaw(tf.keras.models.Model):
    def __init__(self, model, style_layers, preprocess_input):
        super(StyleContentModelRaw, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.preprocess_input = preprocess_input

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = self.preprocess_input(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs = outputs[:len(self.style_layers)]  # Only style layers
        return {style_name: output for style_name, output in zip(self.style_layers, style_outputs)}

# Create a raw extractor for visualization
extractor_raw = StyleContentModelRaw(model,
                                     MODEL_CONFIGS[model_name]['style_layers'],
                                     preprocess_input)

# Visualize raw style layers
def visualize_raw_style_layers(style_image, extractor_raw):
    # Extract raw style layer outputs
    style_outputs = extractor_raw(style_image)

    for layer_name, style_output in style_outputs.items():
        style_output = tf.squeeze(style_output).numpy()  # Remove batch dimension
        num_filters = style_output.shape[-1]
        height, width = style_output.shape[0], style_output.shape[1]

        # Determine grid size
        cols = 8
        rows = num_filters // cols + int(num_filters % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        fig.suptitle(f'Style Layer: {layer_name}', fontsize=16)

        for i in range(num_filters):
            if i >= rows * cols:
                break
            ax = axes[i // cols, i % cols]
            ax.imshow(style_output[..., i], cmap='viridis')
            ax.axis('off')

        # Turn off unused subplots
        for j in range(i + 1, rows * cols):
            axes[j // cols, j % cols].axis('off')

        plt.show()

# Visualize raw style layers
visualize_raw_style_layers(style_image, extractor_raw)

# %%
# Initialize the image to be optimized
image = tf.Variable(content_image)

# Define the optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Function to compute style and content loss
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

# Function to perform a training step
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# %%
# Cell [7]: Training Loop

start_time = time.time()

for n in range(epochs):
    print(f"Epoch {n+1}/{epochs}")
    for m in range(steps_per_epoch):
        train_step(image)
        print('.', end='')
    print()
    # Display intermediate result
    clear_output(wait=True)
    imshow(image.read_value(), title=f"Epoch {n+1}")
    # Optionally save intermediate images
    intermediate_image = tensor_to_image(image)
    intermediate_image.save(f"output_epoch_{n+1}.png")

total_time = time.time() - start_time
print(f"Total time: {total_time:.1f} seconds")


