# Sign Language to Text Converter

This project is a real-time Sign Language to Text converter using computer vision. It captures hand gestures through a webcam and converts them into text, with optional text-to-speech output.

## Features

- Real-time hand gesture detection using MediaPipe
- American Sign Language (ASL) alphabet recognition
- Text-to-speech output for detected signs
- Simple and intuitive user interface
- Webcam support

## Requirements

- Python 3.7+
- Webcam
- Required packages (install using `pip install -r requirements.txt`):
  - OpenCV
  - MediaPipe
  - NumPy
  - pyttsx3
  - Pillow

## Installation

1. Clone this repository:
```bash
git clone <https://github.com/MrShalby/Sign-language-recognizer>
cd sign-language-recognizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Position your hand in front of the webcam
3. Make ASL gestures
4. The detected sign will be displayed on screen and spoken through text-to-speech
5. Press 'q' to quit the application

## Project Structure

- `main.py`: Main application script
- `hand_detector.py`: Hand detection using MediaPipe
- `sign_classifier.py`: Sign language gesture classification
- `text_to_speech.py`: Text-to-speech conversion
- `requirements.txt`: Required Python packages

## Current Limitations

- Basic implementation currently supports a limited set of ASL alphabet signs
- Accuracy may vary based on lighting conditions and hand positioning
- Requires clear view of hand gestures

## Future Improvements

- Add support for more ASL signs
- Implement machine learning for improved accuracy
- Add support for continuous sign language sentences
- Improve gesture recognition accuracy
- Add support for Indian Sign Language (ISL)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 