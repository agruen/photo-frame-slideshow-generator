# Photo Slideshow with Face Detection

A Python script that automatically crops photos to fit your display resolution while keeping faces visible, then generates an HTML slideshow with weather and clock display.

## Features

- 🖼️ **Smart Image Processing**: Automatically resizes and crops images to fit your display
- 👤 **Face Detection**: Uses dlib to ensure faces remain visible when cropping (optional)
- 🌤️ **Weather Integration**: Shows current weather with icons using OpenWeatherMap API
- 🕐 **Live Clock**: Displays current time with automatic updates
- 📱 **Configurable Resolution**: Easy customization for any display size
- ⚡ **Multiprocessing**: Fast parallel image processing
- 🔄 **Random Slideshow**: Images change every 60 seconds

## Prerequisites

### System Requirements
- Python 3.6+
- OpenCV (cv2)
- dlib
- Internet connection (for weather data)

### Dependencies Installation

#### Install Python packages:
```bash
pip install opencv-python dlib
```

#### For macOS users with Homebrew:
```bash
brew install cmake
pip install dlib
```

#### For Ubuntu/Debian users:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

### Face Detection Model (Optional but Recommended)

The script can work without face detection, but for best results, download the dlib face landmarks file:

1. **Download the face landmarks predictor:**
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   ```

2. **Extract the file:**
   ```bash
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

3. **Place the file in the same directory as `crop-and-slideshow.py`**

**Alternative download methods:**
- Direct download: [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Using curl: `curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`

## Setup & Configuration

### 1. Configure the Script

Edit the configuration variables at the top of `crop-and-slideshow.py`:

```python
# Configuration - Edit these settings as needed
IMAGE_DIRECTORY = "."      # Directory containing source images
ZIP_CODE = "10009"         # Your ZIP code for weather data
OUTPUT_FOLDER = "output"   # Subfolder for processed images and HTML

# Screen Resolution Settings
SCREEN_WIDTH = 1280        # Target screen width in pixels
SCREEN_HEIGHT = 800        # Target screen height in pixels
```

### 2. Common Display Resolutions

Update `SCREEN_WIDTH` and `SCREEN_HEIGHT` for your display:

| Display Type | Width | Height | Description |
|--------------|-------|--------|-------------|
| 1280x800 | 1280 | 800 | 16:10 widescreen |
| 1920x1080 | 1920 | 1080 | Full HD (16:9) |
| 1366x768 | 1366 | 768 | Common laptop |
| 1440x900 | 1440 | 900 | 16:10 widescreen |
| 1600x900 | 1600 | 900 | 16:9 widescreen |
| 1024x768 | 1024 | 768 | 4:3 traditional |

### 3. Weather API Setup

The script uses OpenWeatherMap API for weather data. The included API key has usage limits. For production use:

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Replace the API key in the script:
   ```python
   var apiKey = 'YOUR_API_KEY_HERE';
   ```

## Usage

### 1. Prepare Your Images
- Place your photos in the same directory as the script (or specify a different directory in `IMAGE_DIRECTORY`)
- Supported formats: JPG, JPEG, PNG, GIF
- Any resolution - the script will automatically resize and crop

### 2. Run the Script
```bash
python crop-and-slideshow.py
```

### 3. View the Slideshow
- Open `output/index.html` in a web browser
- For fullscreen: Press F11 (most browsers)
- The slideshow will automatically start with weather and clock display

## How It Works

### Image Processing Logic

**With Face Detection:**
1. Resizes images to target width while maintaining aspect ratio
2. Detects faces using dlib's facial landmark detection
3. Crops to target height ensuring all faces remain visible
4. Adds padding around faces for better composition

**Without Face Detection:**
1. Resizes images to target width
2. Crops from center to achieve target height
3. Works well for centered subjects

### File Structure After Running
```
your-project/
├── crop-and-slideshow.py
├── shape_predictor_68_face_landmarks.dat (optional)
├── your-photos.jpg
└── output/
    ├── processed_photo1.jpg
    ├── processed_photo2.jpg
    ├── ...
    └── index.html
```

## Troubleshooting

### Common Issues

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"No module named 'dlib'"**
- On macOS: `brew install cmake && pip install dlib`
- On Ubuntu: `sudo apt-get install build-essential cmake && pip install dlib`
- On Windows: `pip install cmake && pip install dlib`

**Face detection not working:**
- Ensure `shape_predictor_68_face_landmarks.dat` is in the same directory
- The script will still work without it, cropping from center instead

**Weather not loading:**
- Check internet connection
- Verify ZIP_CODE is correct
- API key may have hit rate limits (sign up for your own key)

**Images appear stretched:**
- Verify SCREEN_WIDTH and SCREEN_HEIGHT match your actual display resolution
- Check that your browser is in fullscreen mode

### Performance Tips

- **Large image collections**: The script uses multiprocessing for faster processing
- **Memory usage**: Very large images are automatically resized
- **Processing time**: Expect 1-3 seconds per image depending on size and face detection

## Customization

### Slideshow Timing
Edit these values in the HTML section:
```javascript
}, 60000); // Change slideshow interval (60000 = 60 seconds)
}, 240000); // Change weather update interval (240000 = 4 minutes)
```

### Styling
Modify the CSS in the `generate_slideshow_html()` function to customize:
- Font sizes and colors
- Clock and weather positioning
- Background effects

## License

This project is open source. Feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Ensure all dependencies are installed
3. Verify your configuration settings
4. Open an issue with details about your setup and error messages