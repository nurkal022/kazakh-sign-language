# Kazakh Sign Language Recognition

Real-time recognition system for Kazakh Sign Language using computer vision and deep learning.

## Features

- Real-time hand gesture recognition for 41 letters of the Kazakh alphabet
- Custom dataset collection tool
- ResNet50V2-based deep learning model
- Real-time recognition with confidence levels
- Support for both Latin and Cyrillic character display
- Interactive confidence threshold adjustment
- Multiple visualization modes

## Requirements

- Python 3.8+
- Webcam
- NVIDIA GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kazakh-sign-language.git
cd kazakh-sign-language
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model (optional):
Place the model file in the `Model` directory:
- `Model/kazakh_signs_resnet.3.0.keras`

## Project Structure

```
├── Model/                      # Model files and training artifacts
│   ├── labels.txt             # Class labels
│   └── model_config.json      # Model configuration
├── train_data2/               # Training dataset directory
├── app.py                     # Main application for real-time recognition
├── dtc.py                     # Dataset collection tool
├── train_model.py            # Model training script
├── realtime_recognition.py   # Real-time recognition with advanced features
├── test_model.py            # Model testing utility
└── requirements.txt         # Python dependencies
```

## Usage

### 1. Dataset Collection

To collect your own dataset:
```bash
python dtc.py
```
- Press 'S' to save an image
- Press 'N' to skip to the next letter
- Press 'R' to repeat the last save
- Press 'Q' to quit

### 2. Model Training

To train the model:
```bash
python train_model.py
```

### 3. Real-time Recognition

To run the real-time recognition:
```bash
python realtime_recognition.py
```

Controls:
- Use '+' and '-' to adjust confidence threshold
- Press 'Q' to quit

### 4. Testing

To test the model on specific images:
```bash
python test_model.py
```

## Model Architecture

- Base model: ResNet50V2
- Input size: 224x224x3
- Output: 41 classes (Kazakh alphabet letters)
- Training accuracy: ~91%
- Custom preprocessing pipeline with CLAHE normalization

## Dataset

The dataset consists of hand gestures for 41 letters of the Kazakh alphabet:
- Training samples: 3,280 images
- Validation samples: 820 images
- Total: 4,100 images

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for hand detection
- TensorFlow and Keras for deep learning implementation
- OpenCV for image processing
- CVZone for hand tracking utilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Kalzhanov Nurlykhan - nurkal022@gmail.com

Project Link: https://github.com/nurkal022/kazakh-sign-language 