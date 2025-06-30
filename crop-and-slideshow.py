#!/usr/bin/env python3

# Configuration - Edit these settings as needed
IMAGE_DIRECTORY = "."  # Directory containing source images (current folder by default)
ZIP_CODE = "10001"     # Your ZIP code for weather data
OUTPUT_FOLDER = "output"  # Subfolder where cropped images and HTML will be saved

# Screen Resolution Settings
SCREEN_WIDTH = 1280    # Target screen width in pixels
SCREEN_HEIGHT = 800    # Target screen height in pixels

import cv2
import dlib
import os
import json
from glob import glob
from multiprocessing import Pool, cpu_count

def setup_directories():
    """Create output directory and ensure face landmarks file exists"""
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
    """Find all image files in the specified directory"""
    image_extensions = ['jpg', 'jpeg', 'png', 'gif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(directory, f"*.{ext}"), recursive=False))
        image_files.extend(glob(os.path.join(directory, f"*.{ext.upper()}"), recursive=False))
    
    return image_files

def process_image(args):
    """Process a single image: resize and crop to 1280x800 with face detection"""
    image_file, output_dir, detector, predictor = args
    
    try:
        # Load the image with OpenCV
        image = cv2.imread(image_file)
        if image is None:
            print(f"Could not load image: {image_file}")
            return None
            
        h, w, _ = image.shape
        
        # Target dimensions for display
        target_width = SCREEN_WIDTH
        target_height = SCREEN_HEIGHT
        
        # For landscape images
        if w >= h:
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            resized_image = cv2.resize(image, (new_w, new_h))
            
            # If the resized height is greater than target_height, crop to target_height
            if new_h > target_height:
                if detector is not None:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    
                    if len(faces) > 0:
                        # Get face coordinates
                        face_coords = [(face.top(), face.bottom()) for face in faces]
                        y_min = max(0, min(face[0] for face in face_coords) - 30)
                        y_max = min(new_h, max(face[1] for face in face_coords) + 30)
                        
                        # Adjust to get target_height tall image
                        if y_max - y_min < target_height:
                            diff = target_height - (y_max - y_min)
                            y_min = max(0, y_min - diff//2)
                            y_max = y_min + target_height
                            if y_max > new_h:
                                y_max = new_h
                                y_min = new_h - target_height
                        
                        cropped_image = resized_image[y_min:y_max, :]
                    else:
                        # No faces detected, crop from center
                        y_center = new_h // 2
                        y_min = max(0, y_center - target_height//2)
                        y_max = min(new_h, y_center + target_height//2)
                        cropped_image = resized_image[y_min:y_max, :]
                else:
                    # No face detection available, crop from center
                    y_center = new_h // 2
                    y_min = max(0, y_center - target_height//2)
                    y_max = min(new_h, y_center + target_height//2)
                    cropped_image = resized_image[y_min:y_max, :]
                
                final_image = cropped_image
            else:
                final_image = resized_image
        
        # For portrait images
        else:
            # Resize the image to be target_width wide
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            resized_image = cv2.resize(image, (new_w, new_h))
            
            if detector is not None:
                # Convert the image to grayscale for face detection
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = detector(gray)
                
                if len(faces) > 0:
                    # Get the coordinates of faces
                    face_coords = [(face.top(), face.bottom()) for face in faces]
                    
                    # Calculate the crop boundaries to include all faces
                    y_min = max(0, min(face[0] for face in face_coords) - 30)  
                    y_max = min(new_h, max(face[1] for face in face_coords) + 30)  
                    
                    # Adjust the y_min and y_max to get target_height tall image
                    if y_max - y_min < target_height:
                        diff = target_height - (y_max - y_min)
                        y_min = max(0, y_min - diff//2)
                        y_max = y_min + target_height
                        if y_max > new_h:
                            y_max = new_h
                            y_min = new_h - target_height
                    
                    cropped_image = resized_image[y_min:y_max, :]
                    final_image = cropped_image
                else:
                    # No faces detected, crop from center to target_height
                    y_center = new_h // 2
                    y_min = max(0, y_center - target_height//2)
                    y_max = min(new_h, y_center + target_height//2)
                    cropped_image = resized_image[y_min:y_max, :]
                    final_image = cropped_image
            else:
                # No face detection available, crop from center
                y_center = new_h // 2
                y_min = max(0, y_center - target_height//2)
                y_max = min(new_h, y_center + target_height//2)
                cropped_image = resized_image[y_min:y_max, :]
                final_image = cropped_image
        
        # Save the processed image
        output_filename = f"processed_{os.path.basename(image_file)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, final_image)
        return output_filename
        
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None

def generate_slideshow_html(processed_images, output_dir, zip_code):
    """Generate HTML slideshow optimized for configured display resolution"""
    
    # Convert image paths to relative paths for the HTML
    image_list = [f"./{img}" for img in processed_images if img is not None]
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
    var images = {image_list_json};
    var lastWeatherUpdate = 0;

    function getRandomImage(images) {{
      var index = Math.floor(Math.random() * images.length);
      return images[index];
    }}

    function updateSlideshow(images) {{
      var slideshow = document.getElementById("slideshow");
      slideshow.style.backgroundImage = "url('" + getRandomImage(images) + "')";
    }}

    function updateClock() {{
      var clock = document.getElementById("clock");
      var currentDate = new Date();
      var hours = currentDate.getHours();
      var minutes = currentDate.getMinutes();
      var period = hours >= 12 ? "PM" : "AM";

      // Convert to 12-hour format
      hours = hours % 12;
      hours = hours ? hours : 12;

      // Zero padding for minutes
      minutes = minutes < 10 ? "0" + minutes : minutes;

      clock.innerHTML = hours + ":" + minutes + " " + period;
    }}

    function updateWeather(zipCode) {{
      var weatherDiv = document.getElementById("weather");
      var apiKey = '3b90a894de37f00eed14f6d6d6ac9136';
      var url = `https://api.openweathermap.org/data/2.5/weather?zip=${{zipCode}}&units=imperial&appid=${{apiKey}}`;

      var currentTime = new Date().getTime();
      if (currentTime - lastWeatherUpdate >= 300000) {{
        fetch(url)
          .then(response => response.json())
          .then(data => {{
            var temperature = Math.round(data.main.temp);
            var icon = data.weather[0].icon;
            var iconUrl = `http://openweathermap.org/img/wn/${{icon}}@2x.png`;
            weatherDiv.innerHTML = `<img src="${{iconUrl}}" alt="weather icon"> ${{temperature}}°F`;
            lastWeatherUpdate = currentTime;
          }})
          .catch(error => {{
            weatherDiv.innerHTML = "Weather data not available";
            console.error('Error fetching weather data:', error);
          }});
      }}
    }}

    function startSlideshow(zipCode) {{
      if (images.length > 0) {{
        setInterval(function() {{
          updateSlideshow(images);
        }}, 60000);
      }}
      setInterval(updateClock, 1000);
      setInterval(function() {{
        updateWeather(zipCode);
      }}, 240000);
      updateWeather(zipCode);
      updateClock();
      updateSlideshow(images);
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
    """Main function to process images and generate slideshow"""
    print("Setting up directories...")
    image_dir, output_dir, landmarks_file = setup_directories()
    
    print(f"Looking for images in: {image_dir}")
    image_files = find_image_files(image_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize face detection if landmarks file exists
    detector = None
    predictor = None
    if landmarks_file:
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(landmarks_file)
            print("Face detection enabled")
        except Exception as e:
            print(f"Could not initialize face detection: {e}")
    else:
        print("Face detection disabled - will crop from center")
    
    # Prepare arguments for multiprocessing
    process_args = [(img_file, output_dir, detector, predictor) for img_file in image_files]
    
    print("Processing images...")
    # Use multiprocessing to process images in parallel
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
    html_path = generate_slideshow_html(successful_images, output_dir, ZIP_CODE)
    
    print(f"Complete! Output saved to: {output_dir}")
    print(f"Slideshow HTML: {html_path}")
    print(f"Open {html_path} in a web browser to view the slideshow")

if __name__ == "__main__":
    main()