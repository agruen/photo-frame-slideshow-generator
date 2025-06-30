#!/usr/bin/env python3
"""
Photo Frame Slideshow Generator

This script processes a collection of images to create an optimized slideshow for digital photo frames.
It automatically crops and resizes images to fit a target screen resolution while preserving faces
using face detection technology. The output includes processed images and an HTML slideshow with
weather information and clock display.

Features:
- Intelligent cropping with face detection
- Multi-core parallel processing for fast batch operations
- Responsive HTML slideshow with weather integration
- Support for multiple image formats (JPG, PNG, GIF)
- Configurable screen resolution targeting

Requirements:
- OpenCV (cv2)
- dlib with face landmarks model
- shape_predictor_68_face_landmarks.dat file
- OpenWeatherMap API key (free at openweathermap.org/api)

Setup:
1. Get a free API key from openweathermap.org/api
2. Edit the WEATHER_API_KEY variable at the top of this script
3. Set your ZIP_CODE for local weather

Usage:
    python crop-and-slideshow.py

The script will process all images in the configured directory and create an 'output' folder
containing the processed images and slideshow HTML file.
"""

# Configuration - Edit these settings as needed
IMAGE_DIRECTORY = "."  # Directory containing source images (current folder by default)
ZIP_CODE = "10001"     # Your ZIP code for weather data (US format)
WEATHER_API_KEY = "YOUR_API_KEY_HERE"  # OpenWeatherMap API key (get free key at openweathermap.org/api)
OUTPUT_FOLDER = "output"  # Subfolder where cropped images and HTML will be saved

# Screen Resolution Settings - Configure for your display device
SCREEN_WIDTH = 1280    # Target screen width in pixels
SCREEN_HEIGHT = 800    # Target screen height in pixels

# Face Detection Settings - Configure for better face inclusion
MIN_FACE_PADDING_RATIO = 0.15  # Minimum padding around faces as ratio of face height
MAX_FACE_PADDING_PX = 100      # Maximum padding in pixels to prevent excessive margins
FACE_DEBUG = False              # Set to True to print face detection debugging info

import cv2
import dlib
import os
import json
from glob import glob
from multiprocessing import Pool, cpu_count

def setup_directories():
    """
    Initialize the directory structure and check for required files.
    
    Creates the output directory if it doesn't exist and verifies that the
    dlib face landmarks model file is available for face detection.
    
    Returns:
        tuple: A 3-tuple containing:
            - image_dir (str): Absolute path to the source image directory
            - output_dir (str): Absolute path to the output directory
            - landmarks_file (str or None): Path to face landmarks file, or None if not found
    
    Note:
        If the landmarks file is not found, face detection will be disabled and
        images will be cropped from the center instead.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, IMAGE_DIRECTORY) if IMAGE_DIRECTORY != "." else script_dir
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for face landmarks file
    landmarks_file = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(landmarks_file):
        print("Warning: shape_predictor_68_face_landmarks.dat not found. Face detection will be skipped.")
        return image_dir, output_dir, None
    
    return image_dir, output_dir, landmarks_file

def find_image_files(directory):
    """
    Scan a directory for supported image files.
    
    Searches for common image file extensions in both lowercase and uppercase
    variations to ensure compatibility across different file naming conventions.
    
    Args:
        directory (str): Path to the directory to search for images
    
    Returns:
        list: List of absolute file paths to discovered image files
        
    Supported formats:
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - GIF (.gif)
    """
    image_extensions = ['jpg', 'jpeg', 'png', 'gif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(directory, f"*.{ext}"), recursive=False))
        image_files.extend(glob(os.path.join(directory, f"*.{ext.upper()}"), recursive=False))
    
    return image_files

def process_image(args):
    """
    Process a single image with intelligent cropping and resizing.
    
    This function handles the core image processing pipeline:
    1. Load and validate the image
    2. Determine orientation (landscape vs portrait)
    3. Resize to fit target dimensions while maintaining aspect ratio
    4. Apply intelligent cropping using face detection when available
    5. Save the processed image to the output directory
    
    The cropping algorithm prioritizes keeping faces visible in the final image.
    When faces are detected, the crop boundaries are adjusted to include all faces
    with some padding. If no faces are detected or face detection is unavailable,
    the image is cropped from the center.
    
    Args:
        args (tuple): A 4-tuple containing:
            - image_file (str): Path to the source image file
            - output_dir (str): Directory where processed image will be saved
            - detector: dlib face detector object (or None if unavailable)
            - predictor: dlib shape predictor object (or None if unavailable)
    
    Returns:
        str or None: Filename of the successfully processed image, or None if processing failed
        
    Raises:
        Exception: Various exceptions may occur during image processing, which are caught
                  and logged with the specific image filename for debugging
    """
    image_file, output_dir, detector, predictor = args
    
    try:
        # Load the image with OpenCV (BGR color format)
        image = cv2.imread(image_file)
        if image is None:
            print(f"Could not load image: {image_file}")
            return None
            
        # Get original image dimensions (height, width, channels)
        h, w, _ = image.shape
        
        # Target dimensions for display device
        target_width = SCREEN_WIDTH
        target_height = SCREEN_HEIGHT
        
        # Process landscape images (wider than tall)
        if w >= h:
            # Calculate new dimensions maintaining aspect ratio, fitting to screen width
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            resized_image = cv2.resize(image, (new_w, new_h))
            
            # If resized image is taller than screen, we need to crop vertically
            if new_h > target_height:
                if detector is not None:
                    # Convert to grayscale for dlib face detection (required format)
                    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    
                    if len(faces) > 0:
                        # Extract detailed face information for better cropping
                        face_data = []
                        for face in faces:
                            face_height = face.bottom() - face.top()
                            face_width = face.right() - face.left()
                            # Calculate adaptive padding based on face size
                            padding = min(MAX_FACE_PADDING_PX, int(face_height * MIN_FACE_PADDING_RATIO))
                            face_data.append({
                                'top': face.top(),
                                'bottom': face.bottom(),
                                'left': face.left(),
                                'right': face.right(),
                                'height': face_height,
                                'width': face_width,
                                'padding': padding
                            })
                        
                        if FACE_DEBUG:
                            print(f"Found {len(faces)} faces in {os.path.basename(image_file)}")
                            for i, face in enumerate(face_data):
                                print(f"  Face {i+1}: Y={face['top']}-{face['bottom']}, padding={face['padding']}px")
                        
                        # Calculate crop boundaries that include ALL faces with adequate padding
                        y_min = max(0, min(face['top'] - face['padding'] for face in face_data))
                        y_max = min(new_h, max(face['bottom'] + face['padding'] for face in face_data))
                        
                        # Verify all faces will be included in the final crop
                        faces_in_bounds = True
                        for face in face_data:
                            if face['top'] < y_min or face['bottom'] > y_max:
                                faces_in_bounds = False
                                break
                        
                        if not faces_in_bounds:
                            # Recalculate with minimum padding to ensure all faces fit
                            min_padding = 20  # Absolute minimum padding
                            y_min = max(0, min(face['top'] - min_padding for face in face_data))
                            y_max = min(new_h, max(face['bottom'] + min_padding for face in face_data))
                        
                        # Ensure the crop region is exactly target_height pixels tall
                        current_height = y_max - y_min
                        if current_height < target_height:
                            # Expand the crop region symmetrically to reach target height
                            diff = target_height - current_height
                            y_min = max(0, y_min - diff//2)
                            y_max = y_min + target_height
                            # Handle edge case where expansion goes beyond image bounds
                            if y_max > new_h:
                                y_max = new_h
                                y_min = new_h - target_height
                        elif current_height > target_height:
                            # Crop region is too tall, need to trim while keeping faces
                            # Priority: keep face centers in the crop
                            face_centers = [(face['top'] + face['bottom']) // 2 for face in face_data]
                            center_of_faces = sum(face_centers) // len(face_centers)
                            
                            # Adjust crop to target_height centered on faces
                            y_min = max(0, center_of_faces - target_height//2)
                            y_max = y_min + target_height
                            if y_max > new_h:
                                y_max = new_h
                                y_min = new_h - target_height
                        
                        # Final validation: check if all faces are still in bounds
                        final_faces_in_bounds = all(
                            face['top'] >= y_min and face['bottom'] <= y_max 
                            for face in face_data
                        )
                        
                        if FACE_DEBUG:
                            print(f"  Crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                            print(f"  All faces in bounds: {final_faces_in_bounds}")
                        
                        # Crop the image vertically, keeping full width
                        cropped_image = resized_image[y_min:y_max, :]
                    else:
                        # No faces detected - fall back to center cropping
                        y_center = new_h // 2
                        y_min = max(0, y_center - target_height//2)
                        y_max = min(new_h, y_center + target_height//2)
                        cropped_image = resized_image[y_min:y_max, :]
                else:
                    # Face detection not available - use center cropping
                    y_center = new_h // 2
                    y_min = max(0, y_center - target_height//2)
                    y_max = min(new_h, y_center + target_height//2)
                    cropped_image = resized_image[y_min:y_max, :]
                
                final_image = cropped_image
            else:
                final_image = resized_image
        
        # Process portrait images (taller than wide)
        else:
            # Scale image to fit screen width, maintaining aspect ratio
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            resized_image = cv2.resize(image, (new_w, new_h))
            
            if detector is not None:
                # Convert to grayscale for dlib face detection
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                # Run face detection on the resized image
                faces = detector(gray)
                
                if len(faces) > 0:
                    # Extract detailed face information for better cropping
                    face_data = []
                    for face in faces:
                        face_height = face.bottom() - face.top()
                        face_width = face.right() - face.left()
                        # Calculate adaptive padding based on face size
                        padding = min(MAX_FACE_PADDING_PX, int(face_height * MIN_FACE_PADDING_RATIO))
                        face_data.append({
                            'top': face.top(),
                            'bottom': face.bottom(),
                            'left': face.left(),
                            'right': face.right(),
                            'height': face_height,
                            'width': face_width,
                            'padding': padding
                        })
                    
                    if FACE_DEBUG:
                        print(f"Found {len(faces)} faces in {os.path.basename(image_file)} (portrait)")
                        for i, face in enumerate(face_data):
                            print(f"  Face {i+1}: Y={face['top']}-{face['bottom']}, padding={face['padding']}px")
                    
                    # Calculate crop boundaries that include ALL faces with adequate padding
                    y_min = max(0, min(face['top'] - face['padding'] for face in face_data))
                    y_max = min(new_h, max(face['bottom'] + face['padding'] for face in face_data))
                    
                    # Verify all faces will be included in the final crop
                    faces_in_bounds = True
                    for face in face_data:
                        if face['top'] < y_min or face['bottom'] > y_max:
                            faces_in_bounds = False
                            break
                    
                    if not faces_in_bounds:
                        # Recalculate with minimum padding to ensure all faces fit
                        min_padding = 20  # Absolute minimum padding
                        y_min = max(0, min(face['top'] - min_padding for face in face_data))
                        y_max = min(new_h, max(face['bottom'] + min_padding for face in face_data))
                    
                    # Ensure final crop is exactly target_height pixels tall
                    current_height = y_max - y_min
                    if current_height < target_height:
                        # Expand crop region to meet target height requirement
                        diff = target_height - current_height
                        y_min = max(0, y_min - diff//2)
                        y_max = y_min + target_height
                        # Adjust if expansion exceeds image boundaries
                        if y_max > new_h:
                            y_max = new_h
                            y_min = new_h - target_height
                    elif current_height > target_height:
                        # Crop region is too tall, need to trim while keeping faces
                        # Priority: keep face centers in the crop
                        face_centers = [(face['top'] + face['bottom']) // 2 for face in face_data]
                        center_of_faces = sum(face_centers) // len(face_centers)
                        
                        # Adjust crop to target_height centered on faces
                        y_min = max(0, center_of_faces - target_height//2)
                        y_max = y_min + target_height
                        if y_max > new_h:
                            y_max = new_h
                            y_min = new_h - target_height
                    
                    # Final validation: check if all faces are still in bounds
                    final_faces_in_bounds = all(
                        face['top'] >= y_min and face['bottom'] <= y_max 
                        for face in face_data
                    )
                    
                    if FACE_DEBUG:
                        print(f"  Crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                        print(f"  All faces in bounds: {final_faces_in_bounds}")
                    
                    # Apply the calculated crop boundaries
                    cropped_image = resized_image[y_min:y_max, :]
                    final_image = cropped_image
                else:
                    # No faces found - use center-based cropping strategy
                    y_center = new_h // 2
                    y_min = max(0, y_center - target_height//2)
                    y_max = min(new_h, y_center + target_height//2)
                    cropped_image = resized_image[y_min:y_max, :]
                    final_image = cropped_image
            else:
                # Face detection unavailable - default to center cropping
                y_center = new_h // 2
                y_min = max(0, y_center - target_height//2)
                y_max = min(new_h, y_center + target_height//2)
                cropped_image = resized_image[y_min:y_max, :]
                final_image = cropped_image
        
        # Save the processed image with descriptive filename prefix
        output_filename = f"processed_{os.path.basename(image_file)}"
        output_path = os.path.join(output_dir, output_filename)
        # Write image using OpenCV (maintains quality and supports various formats)
        cv2.imwrite(output_path, final_image)
        return output_filename
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None

def generate_slideshow_html(processed_images, output_dir, zip_code, api_key):
    """
    Generate a full-screen HTML slideshow with weather and time display.
    
    Creates a responsive HTML file that displays processed images in a slideshow format
    with integrated weather information and digital clock. The slideshow is optimized
    for the configured screen resolution and includes the following features:
    
    - Random image rotation every 60 seconds
    - Real-time clock display (12-hour format)
    - Weather information with icons and temperature
    - Full-screen display optimized for digital photo frames
    - Automatic weather data refresh every 4 minutes
    
    Args:
        processed_images (list): List of processed image filenames to include in slideshow
        output_dir (str): Directory where the HTML file will be saved
        zip_code (str): US ZIP code for weather data retrieval
        api_key (str): OpenWeatherMap API key for weather data access
    
    Returns:
        str: Absolute path to the generated HTML slideshow file
        
    Note:
        The weather functionality requires an internet connection and uses the
        OpenWeatherMap API. The API key is embedded in the HTML for demonstration
        purposes - in production, consider using environment variables or a backend service.
    """
    
    # Convert image filenames to relative paths for HTML slideshow
    # Filter out any None values from failed image processing
    image_list = [f"./{img}" for img in processed_images if img is not None]
    # Convert to JSON format for embedding in JavaScript
    image_list_json = json.dumps(image_list)
    
    html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Photo Slideshow - {SCREEN_WIDTH}x{SCREEN_HEIGHT}</title>
  <style>
    body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        width: {SCREEN_WIDTH}px;
        height: {SCREEN_HEIGHT}px;
    }}

    #slideshow {{
        width: {SCREEN_WIDTH}px;
        height: {SCREEN_HEIGHT}px;
        position: fixed;
        top: 0;
        left: 0;
        background-color: lightgray;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        z-index: -1;
    }}

    #info {{
        position: absolute;
        bottom: 30px;
        left: 30px;
        width: calc(100% - 60px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: Arial, sans-serif;
        font-size: 64px;
        font-weight: bold;
        color: white;
        z-index: 100;
        text-align: center;
        text-shadow: 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000, -2px 0 0 #000;
    }}

    #clock {{
        text-align: left;
    }}

    #weather {{
        text-align: right;
    }}

    #weather img {{
        vertical-align: middle;
        width: 80px;
        height: 80px;
    }}
  </style>
</head>
<body>
  <div id="slideshow"></div>
  <div id="info">
    <div id="clock"></div>
    <div id="weather"></div>
  </div>

  <script>
    // Array of processed image paths for slideshow rotation
    var images = {image_list_json};
    // Track when weather was last updated (rate limiting)
    var lastWeatherUpdate = 0;

    // Select a random image from the available collection
    function getRandomImage(images) {{
      var index = Math.floor(Math.random() * images.length);
      return images[index];
    }}

    // Update the background image with a random selection
    function updateSlideshow(images) {{
      var slideshow = document.getElementById("slideshow");
      slideshow.style.backgroundImage = "url('" + getRandomImage(images) + "')";
    }}

    // Update the digital clock display with current time
    function updateClock() {{
      var clock = document.getElementById("clock");
      var currentDate = new Date();
      var hours = currentDate.getHours();
      var minutes = currentDate.getMinutes();
      var period = hours >= 12 ? "PM" : "AM";

      // Convert from 24-hour to 12-hour format
      hours = hours % 12;
      hours = hours ? hours : 12;  // Handle midnight (0 hours = 12 AM)

      // Add leading zero to minutes for consistent formatting
      minutes = minutes < 10 ? "0" + minutes : minutes;

      // Display formatted time
      clock.innerHTML = hours + ":" + minutes + " " + period;
    }}

    // Fetch and display current weather information
    function updateWeather(zipCode) {{
      var weatherDiv = document.getElementById("weather");
      // OpenWeatherMap API configuration
      var apiKey = '{api_key}';
      var url = `https://api.openweathermap.org/data/2.5/weather?zip=${{zipCode}}&units=imperial&appid=${{apiKey}}`;

      var currentTime = new Date().getTime();
      // Only update weather every 5 minutes (300000ms) to avoid rate limiting
      if (currentTime - lastWeatherUpdate >= 300000) {{
        fetch(url)
          .then(response => response.json())
          .then(data => {{
            // Extract temperature and weather icon from API response
            var temperature = Math.round(data.main.temp);
            var icon = data.weather[0].icon;
            var iconUrl = `http://openweathermap.org/img/wn/${{icon}}@2x.png`;
            // Display weather icon and temperature
            weatherDiv.innerHTML = `<img src="${{iconUrl}}" alt="weather icon"> ${{temperature}}°F`;
            lastWeatherUpdate = currentTime;
          }})
          .catch(error => {{
            // Handle API errors gracefully
            weatherDiv.innerHTML = "Weather data not available";
            console.error('Error fetching weather data:', error);
          }});
      }}
    }}

    // Initialize and start the slideshow with all interactive elements
    function startSlideshow(zipCode) {{
      // Only start image rotation if we have images to display
      if (images.length > 0) {{
        // Change slideshow image every 60 seconds (60000ms)
        setInterval(function() {{
          updateSlideshow(images);
        }}, 60000);
      }}
      // Update clock every second for real-time display
      setInterval(updateClock, 1000);
      // Check for weather updates every 4 minutes (240000ms)
      setInterval(function() {{
        updateWeather(zipCode);
      }}, 240000);
      
      // Initialize display elements immediately
      updateWeather(zipCode);   // Get initial weather data
      updateClock();           // Show current time
      updateSlideshow(images); // Display first image
    }}

    startSlideshow('{zip_code}');
  </script>
</body>
</html>"""

    # Write the HTML file to the output directory
    html_path = os.path.join(output_dir, 'index.html')
    with open(html_path, 'w') as file:
        file.write(html_template)
    
    return html_path

def main():
    """
    Main execution function that orchestrates the entire slideshow generation process.
    
    This function coordinates all the major steps:
    1. Set up directories and check for required files
    2. Discover all image files in the source directory
    3. Initialize face detection capabilities if available
    4. Process all images in parallel using multiprocessing
    5. Generate the HTML slideshow with processed images
    6. Provide user feedback and final instructions
    
    The function uses all available CPU cores for parallel image processing to
    minimize processing time for large image collections.
    
    Returns:
        None
        
    Side Effects:
        - Creates output directory structure
        - Processes and saves resized/cropped images
        - Generates HTML slideshow file
        - Prints progress information to console
    """
    print("Setting up directories...")
    image_dir, output_dir, landmarks_file = setup_directories()
    
    print(f"Looking for images in: {image_dir}")
    image_files = find_image_files(image_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize face detection components if the required model file exists
    detector = None
    predictor = None  # Shape predictor for detailed facial landmarks (not used in current implementation)
    if landmarks_file:
        try:
            # Load dlib's pre-trained face detector (HOG + Linear SVM)
            detector = dlib.get_frontal_face_detector()
            # Load the 68-point facial landmark predictor model
            predictor = dlib.shape_predictor(landmarks_file)
            print("Face detection enabled")
        except Exception as e:
            print(f"Could not initialize face detection: {e}")
            print("Falling back to center-based cropping")
    else:
        print("Face detection disabled - will crop from center")
    
    # Prepare argument tuples for parallel processing
    # Each worker process needs: (image_path, output_dir, face_detector, landmark_predictor)
    process_args = [(img_file, output_dir, detector, predictor) for img_file in image_files]
    
    print("Processing images...")
    # Utilize all available CPU cores for parallel image processing
    # This significantly reduces processing time for large image collections
    with Pool(cpu_count()) as pool:
        processed_images = pool.map(process_image, process_args)
    
    # Filter out None results (failed processing)
    successful_images = [img for img in processed_images if img is not None]
    
    if not successful_images:
        print("No images were successfully processed!")
        return
    
    print(f"Successfully processed {len(successful_images)} images")
    
    # Generate HTML slideshow
    print("Generating slideshow HTML...")
    html_path = generate_slideshow_html(successful_images, output_dir, ZIP_CODE, WEATHER_API_KEY)
    
    print(f"Complete! Output saved to: {output_dir}")
    print(f"Slideshow HTML: {html_path}")
    print(f"Open {html_path} in a web browser to view the slideshow")

if __name__ == "__main__":
    main()