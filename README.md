## Overview

This script implements **Neural Style Transfer** using various pre-trained deep learning models available in Keras Applications. It blends the artistic style of one image (the "style image") with the content of another (the "content image"), producing a visually compelling combination.

Inspired by [TensorFlow's DeepDream Tutorial](https://www.tensorflow.org/tutorials/generative/deepdream).

## Example Results

### Input Images

**Content Image**: Mac Miller - Tiny Desk Concert
![download](https://github.com/user-attachments/assets/6bb0cc87-bf54-420b-9003-34360ff28df8)

**Style Image**: Duncan Jago - Scop(2024)
![download](https://github.com/user-attachments/assets/0cf27a84-999d-4b6c-b736-299ffce4015c)

### Style Transfer Process

**Style Layer Visualization (block1_conv1)**:
![image](https://github.com/user-attachments/assets/c8e9de43-1a23-49fb-861b-90f53ab5100b)

**Final Result**:
After 10 epochs of style transfer (10 steps per epoch)
![download](https://github.com/user-attachments/assets/26ec3519-34c5-40fd-a478-7a8833ecddd9)

## Features

- **Multiple Model Support**: VGG16, VGG19, InceptionV3, Xception, ResNet, MobileNet, DenseNet, and EfficientNet variants
- **Customizable Parameters**: Fine-tune content weight, style weight, and total variation weight
- **Progress Tracking**: Visualize and save intermediate results during the style transfer process
- **Flexible Integration**: Compatible with Jupyter Notebooks, Google Colab, and standalone Python scripts

## Requirements

- TensorFlow 2.x
- Python 3.x
- NumPy
- Pillow (PIL)
- Matplotlib
- IPython

## Quick Start

1. Clone the repository:
```bash
git clone [repository-url]
cd style_transfer_vgg19
```

2. Install dependencies:
```bash
pip install tensorflow numpy pillow matplotlib ipython
```

3. Configure your image paths and parameters in `style_transfer.py`:
```python
content_image_path = 'path/to/your/content/image.jpg'
style_image_path = 'path/to/your/style/image.jpg'

# Adjust weights as needed
content_weight = 1e4
style_weight = 1e-2
total_variation_weight = 30

# Training parameters
epochs = 10
steps_per_epoch = 20
max_dim = 512
```

4. Run the script:
```bash
python style_transfer.py
```

## Configuration Options

### Available Models
- VGG16/VGG19 (recommended for best results)
- ResNet (50/101/152) and V2 variants
- Inception V3
- MobileNet/MobileNetV2
- DenseNet (121/169/201)
- EfficientNet (B0-B7, V2)

### Key Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `content_weight` | Controls content preservation | 1e4 |
| `style_weight` | Controls style intensity | 1e-2 |
| `total_variation_weight` | Controls image smoothness | 30 |
| `max_dim` | Maximum image dimension | 512 |
| `epochs` | Number of training epochs | 10 |
| `steps_per_epoch` | Steps per epoch | 20 |

## Tips for Best Results

1. **Image Selection**:
   - Choose style images with strong, distinctive patterns
   - Ensure content images have good contrast and clear subjects

2. **Parameter Tuning**:
   - Increase `style_weight` for stronger stylization
   - Increase `content_weight` to preserve more original content
   - Adjust `total_variation_weight` to control noise/smoothness

3. **Model Selection**:
   - VGG models work best for traditional artistic styles
   - Try ResNet or EfficientNet for modern/abstract styles

## Troubleshooting

- **Memory Issues**: Reduce `max_dim` or batch size
- **Poor Results**: Try adjusting weights or changing the model
- **Slow Processing**: Use a smaller image size or fewer epochs

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Based on TensorFlow's Neural Style Transfer implementation
- Inspired by the paper "A Neural Algorithm of Artistic Style" by Gatys et al.

