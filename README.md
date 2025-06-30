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

#### For macOS Users (Recommended Setup with Virtual Environment)

Modern macOS requires using Python virtual environments. Follow these steps:

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv photo-slideshow-env
   source photo-slideshow-env/bin/activate
   ```

2. **Install system dependencies:**
   ```bash
   brew install cmake
   ```

3. **Install Python packages:**
   ```bash
   pip install opencv-python dlib
   ```

4. **To run the script later, always activate the environment first:**
   ```bash
   source photo-slideshow-env/bin/activate
   python crop-and-slideshow.py
   ```

#### For Other Platforms:

**General Python packages:**
```bash
pip install opencv-python dlib
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
WEATHER_API_KEY = "YOUR_API_KEY_HERE"  # OpenWeatherMap API key
OUTPUT_FOLDER = "output"   # Subfolder for processed images and HTML

# Screen Resolution Settings
SCREEN_WIDTH = 1280        # Target screen width in pixels
SCREEN_HEIGHT = 800        # Target screen height in pixels

# Face Detection Settings
MIN_FACE_PADDING_RATIO = 0.15  # Minimum padding around faces (15% of face height)
MAX_FACE_PADDING_PX = 100      # Maximum padding in pixels
FACE_DEBUG = False              # Set to True to see face detection details
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

The script uses OpenWeatherMap API for weather data.

**⚠️ API Key Required:**
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api) for a free API key
2. Edit the configuration in `crop-and-slideshow.py`:
   ```python
   WEATHER_API_KEY = "YOUR_API_KEY_HERE"
   ```

**Rate Limiting:**
The weather updates every 4 minutes (240 seconds) to stay within the free API limits:
- Free tier: 1,000 calls/day
- 4-minute intervals = ~360 calls/day
- Leaves room for initial loads and testing

To change the update frequency, modify the interval in the generated HTML, but be mindful of API limits.

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

### Advanced Face Detection

The script includes intelligent face detection that ensures all faces in group photos remain visible:

**Enhanced Face Detection Features:**
- **Adaptive Padding**: Padding scales with face size (15% of face height, max 100px)
- **Multiple Face Support**: Detects and includes ALL faces in the crop area
- **Smart Centering**: When cropping is needed, centers on the collective center of all faces
- **Validation**: Verifies all faces remain within final crop boundaries

**Face Debug Mode:**
To troubleshoot face detection, enable debug mode:
```python
FACE_DEBUG = True
```

This will show detailed output like:
```
Found 3 faces in family_photo.jpg
  Face 1: Y=120-180, padding=25px
  Face 2: Y=140-200, padding=30px  
  Face 3: Y=110-170, padding=28px
  Crop region: Y=85-685 (height=800)
  All faces in bounds: True
```

**Processing Logic:**

**With Face Detection:**
1. Resizes images to target width while maintaining aspect ratio
2. Detects all faces using dlib's facial landmark detection
3. Calculates adaptive padding for each face based on size
4. Creates crop boundaries that include all faces with padding
5. Validates all faces remain visible in final crop
6. Centers crop on faces if adjustment is needed

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
- Enable `FACE_DEBUG = True` to see what faces are being detected
- Check that images have clear, front-facing faces for best detection

**Weather not loading:**
- Check internet connection
- Verify ZIP_CODE is correct (US format like "10001")
- Ensure WEATHER_API_KEY is set to your OpenWeatherMap API key
- API key may have hit daily rate limits (free tier = 1,000 calls/day)
- Check browser console for API error messages

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
}, 60000);  // Change slideshow interval (60000 = 60 seconds)
}, 240000); // Weather update interval (240000 = 4 minutes)
             // ⚠️ Don't reduce below 4 minutes to avoid exceeding API limits
```

### Face Detection Customization
Adjust face detection behavior in the configuration:
```python
MIN_FACE_PADDING_RATIO = 0.15  # Increase for more padding around faces
MAX_FACE_PADDING_PX = 100      # Maximum padding to prevent excessive margins
FACE_DEBUG = True               # Enable to see face detection details
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