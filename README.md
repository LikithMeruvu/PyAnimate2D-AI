# PyAnimate2D-AI 🎨

A powerful AI-powered animation generation system that combines Flask, Pygame, and Google's Gemini AI to create dynamic 2D animations from text descriptions.

## 🌟 Features

- **AI-Powered Animation Generation**: Leverages Google's Gemini AI to convert text descriptions into Python animation code
- **Real-Time Preview**: Instantly view generated animations in the browser
- **Customizable Settings**: Control animation duration, resolution, and frame rate
- **Code Visibility**: View and learn from the AI-generated Python code
- **Video Export**: Download animations in MP4 format
- **History Management**: Keep track of previous generations and chat history
- **Responsive UI**: Modern, user-friendly interface built with Tailwind CSS


(https://github.com/user-attachments/assets/ff500c8b-a7ee-48ac-8529-bc7f71a2a4ce)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Modern web browser
- FFmpeg (for video processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LikithMeruvu/PyAnimate2D-AI.git
cd PyAnimate2D-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your environment:
   - Obtain a Google Gemini API key
   - Have FFmpeg installed and accessible in your system PATH

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## 🎮 Usage

### Configuration

1. Enter your Gemini API key in the configuration section
2. Select your preferred Gemini model
3. Adjust the temperature setting (0.0-2.0)
4. Save your configuration

### Generating Animations

1. Enter a descriptive prompt for your desired animation
2. Set animation parameters:
   - Duration (in seconds, max 60)
   - Resolution (144p to 1440p)
   - Frame rate (15-60 FPS)
3. Click "Generate Animation"
4. Wait for the generation process to complete
5. Preview and download your animation

## 🛠️ Technical Details

### Architecture

- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Backend**: Flask (Python)
- **Animation Engine**: Pygame
- **AI Integration**: Google Generative AI (Gemini)
- **Video Processing**: OpenCV, FFmpeg

### Key Components

1. **Flask Application (`app.py`)**
   - Handles HTTP routes and requests
   - Manages session data
   - Coordinates between UI and animation generation
   - Handles video file management

2. **Animation Generation**
   - Uses Pygame for rendering
   - Converts AI-generated code into animations
   - Supports custom resolutions and frame rates
   - Handles error recovery and retries

3. **User Interface**
   - Responsive design with Tailwind CSS
   - Real-time video preview with Video.js
   - Code syntax highlighting with Prism.js
   - Interactive chat history

4. **Video Processing**
   - OpenCV for frame processing
   - FFmpeg for video encoding
   - Supports multiple video formats and qualities

## 🎨 Animation Capabilities

The system can generate various types of 2D animations, including:
- Geometric patterns and shapes
- Particle systems
- Character animations
- Abstract visualizations
- Mathematical animations
- Interactive scenes

## 🔧 Configuration Options

### Video Settings
- **Resolution**: 144p to 1440p
- **Frame Rate**: 15, 30, 45, or 60 FPS
- **Duration**: Up to 60 seconds
- **Format**: MP4 (H.264 codec)

### AI Settings
- **Model Selection**: Multiple Gemini models available
- **Temperature**: 0.0 to 2.0 (controls creativity)
- **API Configuration**: Custom API key support

## 📝 Development Guidelines

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Document new functions and classes
- Add appropriate error handling

## 🐛 Troubleshooting

### Common Issues

1. **Video Generation Fails**
   - Check FFmpeg installation
   - Verify write permissions in output directories
   - Check available system memory

2. **AI Generation Errors**
   - Verify API key validity
   - Check internet connection
   - Review prompt complexity

3. **Performance Issues**
   - Lower video resolution
   - Reduce animation duration
   - Check system resources

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 🙏 Acknowledgments

- Google Generative AI team for the Gemini API
- Pygame community for the animation framework
- Flask team for the web framework
- Contributors and users of the project

## 📞 Support

For support, please:
- Open an issue on GitHub
- Join our Discord community
- Check the documentation wiki

---

Made with ❤️ by the PyAnimate2D-AI team
