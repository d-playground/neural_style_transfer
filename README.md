# Neural Style Transfer with VGG19

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A TensorFlow implementation of neural style transfer using VGG19 and other pre-trained models.

## Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Technical Details](#technical-details)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This script implements **Neural Style Transfer** using various pre-trained deep learning models available in Keras Applications. It blends the artistic style of one image (the "style image") with the content of another (the "content image"), producing a visually compelling combination.

Inspired by [TensorFlow's DeepDream Tutorial](https://www.tensorflow.org/tutorials/generative/deepdream).

## Demo

### Before and After Style Transfer

<table>
<tr>
    <td><b>Content Image</b>: Mac Miller - Tiny Desk Concert</td>
    <td><b>Style Image</b>: Duncan Jago - Scop(2024)</td>
</tr>
<tr>
    <td><img src="https://github.com/user-attachments/assets/6bb0cc87-bf54-420b-9003-34360ff28df8" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/0cf27a84-999d-4b6c-b736-299ffce4015c" width="400"/></td>
</tr>
<tr>
    <td colspan="2" align="center"><b>Result after Style Transfer</b></td>
</tr>
<tr>
    <td colspan="2" align="center"><img src="https://github.com/user-attachments/assets/26ec3519-34c5-40fd-a478-7a8833ecddd9" width="600"/></td>
</tr>
</table>

## Installation

### Prerequisites
- Python 3.x
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository
```bash
git clone [repository-url]
cd style_transfer_vgg19
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your images:
   - Place your content image and style image in a known directory
   - Supported formats: JPG, PNG

2. Configure parameters in `style_transfer.py`:
```python
content_image_path = 'path/to/your/content/image.jpg'
style_image_path = 'path/to/your/style/image.jpg'

# Adjust weights as needed
content_weight = 1e4      # Controls content preservation
style_weight = 1e-2       # Controls style intensity
total_variation_weight = 30  # Controls smoothness
```

3. Run the script:
```bash
python style_transfer.py
```

## Models

### Supported Architectures
- VGG16/VGG19 (recommended)
- ResNet (50/101/152)
- Inception V3
- MobileNet/V2
- DenseNet
- EfficientNet

### Model Selection Guide
| Model | Best Use Case | Memory Usage | Speed |
|-------|--------------|--------------|--------|
| VGG19 | Traditional art styles | High | Medium |
| ResNet | Modern/abstract styles | Medium | Fast |
| MobileNet | Resource-constrained environments | Low | Very Fast |

## Technical Details

### Architecture Overview
```
Input Image → Pre-trained CNN → Feature Maps → Style/Content Loss → Optimization → Styled Image
```

### Loss Functions
1. **Content Loss**: `L_content = content_weight * mean((content_features - generated_features)²)`
2. **Style Loss**: `L_style = style_weight * mean((style_gram - generated_gram)²)`
3. **Total Loss**: `L_total = L_content + L_style + total_variation_weight * L_tv`

### Optimization
- Optimizer: Adam
- Learning Rate: 0.02
- Training Steps: epochs * steps_per_epoch

## Performance

### Resource Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8GB | 16GB |
| GPU VRAM | 4GB | 8GB+ |
| Storage | 1GB | 5GB |

### Processing Time
| Image Size | GPU Time | CPU Time |
|------------|----------|----------|
| 512px | 1-2 min | 5-10 min |
| 1024px | 3-5 min | 15-30 min |
| 2048px | 8-12 min | 45-90 min |

## Troubleshooting

### Common Issues
1. **Out of Memory**
   - Reduce image dimensions
   - Lower batch size
   - Use lighter model

2. **Poor Results**
   - Adjust style weight
   - Try different model
   - Check image quality

3. **Slow Processing**
   - Enable GPU acceleration
   - Reduce image size
   - Decrease iteration count

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on TensorFlow's Neural Style Transfer implementation
- Inspired by ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al.
- Thanks to the TensorFlow team for their excellent tutorials and documentation

