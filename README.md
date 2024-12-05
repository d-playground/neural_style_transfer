![image](https://github.com/user-attachments/assets/f54b88ce-3b40-4763-9988-aa8db814e16f)## Overview
This script implements a **Neural Style Transfer** process using pre-trained deep learning models (VGG16, VGG19, and InceptionV3) available in TensorFlow. It allows the user to blend the artistic style of one image with the content of another, creating a visually compelling combination.

This project is inspired by [Tensorflow Tutorial: DeepDream]([url](https://www.tensorflow.org/tutorials/generative/deepdream))




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

- Supports three pre-trained models: **VGG16**, **VGG19**, and **InceptionV3**.
- Customizable parameters for fine-tuning the style transfer process.
- Generates intermediate results for visualization after every epoch.
- Saves the final stylized image after training.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow
- Matplotlib
- IPython (for displaying images during execution)

## Installation

1. Clone the repository or download the script file.
2. Install the required libraries:

   ```bash
   pip install tensorflow numpy pillow matplotlib ipython
   ```

3. Ensure you have access to your content and style images and update their paths in the script.

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

- **Model Selection**: Change `model_name` to `'VGG16'`, `'VGG19'`, or `'InceptionV3'` to use different feature extractors.
- **Layers for Style/Content**: Modify `style_layers` and `content_layers` for finer control of the transfer process.
- **Learning Rate**: Adjust the optimizer’s learning rate for faster or more stable convergence.

## Notes

- Ensure your image paths are valid. The script uses assertions to check if the images exist.
- Colab users must mount their Google Drive to access images.
- Experiment with weights to achieve desired results; higher `style_weight` enhances the style effect, while higher `content_weight` retains more of the original content.

