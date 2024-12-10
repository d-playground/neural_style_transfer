## Overview

This script implements **Neural Style Transfer** using various pre-trained deep learning models available in Keras Applications. It blends the artistic style of one image (the "style image") with the content of another (the "content image"), producing a visually compelling combination.

Inspired by [TensorFlow's DeepDream Tutorial](https://www.tensorflow.org/tutorials/generative/deepdream).





## Example: 

### Input: 

*Content Image: Mac Miller - Tiny Desk Concert*

![download](https://github.com/user-attachments/assets/6bb0cc87-bf54-420b-9003-34360ff28df8)


*Style Image: Duncan Jago - Scop(2024)*

![download](https://github.com/user-attachments/assets/0cf27a84-999d-4b6c-b736-299ffce4015c)



### Extracted Style Image Layers (block1_conv1): 
![image](https://github.com/user-attachments/assets/c8e9de43-1a23-49fb-861b-90f53ab5100b)



### Output: 
Content image after 10 epochs of style transfer (10 steps per epoch).
![download](https://github.com/user-attachments/assets/26ec3519-34c5-40fd-a478-7a8833ecddd9)




## Features

- **Multiple Pre-Trained Models**: Choose from various models including **VGG16**, **VGG19**, **InceptionV3**, **Xception**, **ResNet**, **MobileNet**, **DenseNet**, and **EfficientNet** variants.
- **Customizable Parameters**: Adjust weights for content, style, and total variation. Control training duration, image size, and learning rate.
- **Intermediate Results**: Save and visualize intermediate outputs for inspection.
- **Easy Integration**: Straightforward to run in Jupyter Notebooks, Colab, or Python scripts.

## Supported Models

You can choose from the following supported models:
'Xception' 'VGG16' 'VGG19' 'ResNet50' 'ResNet50V2' 'ResNet101' 'ResNet101V2' 'ResNet152' 'ResNet152V2' 'InceptionV3' 'MobileNet' 'MobileNetV2' 'DenseNet121' 'DenseNet169' 'DenseNet201' 'EfficientNetB0' 'EfficientNetB1' 'EfficientNetB2' 'EfficientNetB3' 'EfficientNetB4' 'EfficientNetB5' 'EfficientNetB6' 'EfficientNetB7' 'EfficientNetV2B0' 'EfficientNetV2B1' 'EfficientNetV2B2' 'EfficientNetV2B3' 'EfficientNetV2S' 'EfficientNetV2M' 'EfficientNetV2L'

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow
- Matplotlib
- IPython (for displaying images during execution)

## Installation

1. Clone the repository to google colab.
2. adjust the path of the content and style images.
3. adjust parameters in the script.
4. run the script.


## Usage

### Running the Script

1. Open the script in your preferred Python environment (e.g., Jupyter Notebook, Colab, or directly in a terminal).
2. Mount your Google Drive if using Colab and update the paths to your content and style images.
3. Adjust parameters in the `Parameters` section as desired:
   - `content_weight`: Weight for preserving the content features.
   - `style_weight`: Weight for stylizing the image.
   - `total_variation_weight`: Weight for smoothing the generated image.
   - `epochs` and `steps_per_epoch`: Control the training duration.
   - `max_dim`: Maximum dimension for resizing the input images.
4. Run the script. Intermediate results will be displayed during execution.

### Example Parameters

```python
# Example Parameters
model_name = 'VGG16'  # Choose: 'VGG16', 'VGG19', 'InceptionV3'
content_image_path = 'path/to/content_image.jpg'
style_image_path = 'path/to/style_image.jpg'
content_weight = 1e4
style_weight = 1e-1
total_variation_weight = 30
epochs = 10
steps_per_epoch = 20
max_dim = 512
```

### Output

The script saves intermediate images during training in the current working directory with filenames like `output_epoch_N.png`. The final stylized image is displayed and can be saved manually or automatically.

## Customization

- **Model Layers**: Modify the `style_layers` and `content_layers` chosen from the selected model for more fine-grained control.
- **Hyperparameters**: Adjust `content_weight`, `style_weight`, and `total_variation_weight` to achieve different effects.
- **Learning Rate**: Tune the optimizer’s learning rate for more stable or faster convergence.

## Notes

- Ensure your image paths are valid. The script uses assertions to check if the images exist.
- Colab users must mount their Google Drive to access images.
- Experiment with weights to achieve desired results; higher `style_weight` enhances the style effect, while higher `content_weight` retains more of the original content.

