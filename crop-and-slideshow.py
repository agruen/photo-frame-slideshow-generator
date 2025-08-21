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
- MediaPipe (for advanced face detection)
- NumPy
- OpenWeatherMap API key (free at openweathermap.org/api)

Setup:
1. Install required libraries: pip install opencv-python mediapipe numpy
2. Get a free API key from openweathermap.org/api
3. Edit the WEATHER_API_KEY variable at the top of this script
4. Set your ZIP_CODE for local weather

Usage:
    python crop-and-slideshow.py

The script will process all images in the configured directory and create an 'output' folder
containing the processed images and slideshow HTML file.
"""

import cv2
import numpy as np
import os
import json
from glob import glob
from multiprocessing import Pool, cpu_count
from retinaface import RetinaFace
from ultralytics import YOLO

# Configuration - Edit these settings as needed or use environment variables
IMAGE_DIRECTORY = os.getenv("IMAGE_DIRECTORY", ".")  # Directory containing source images
ZIP_CODE = os.getenv("ZIP_CODE", "10001")     # Your ZIP code for weather data (US format)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "YOUR_API_KEY_HERE")  # OpenWeatherMap API key
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "output")  # Subfolder where cropped images and HTML will be saved

# Screen Resolution Settings - Configure for your display device
SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH", "1280"))    # Target screen width in pixels
SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT", "800"))    # Target screen height in pixels

# Face Detection Settings - Configure for better face inclusion
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.7"))       # Minimum confidence score for face detection (0.0-1.0)
FACE_PADDING_RATIO = float(os.getenv("FACE_PADDING_RATIO", "0.2"))        # Padding around faces as ratio of face size
MAX_FACE_PADDING_PX = int(os.getenv("MAX_FACE_PADDING_PX", "120"))       # Maximum padding in pixels to prevent excessive margins
FACE_SIZE_THRESHOLD = float(os.getenv("FACE_SIZE_THRESHOLD", "0.02"))      # Minimum face size as ratio of image area
FACE_DEBUG = os.getenv("FACE_DEBUG", "false").lower() == "true"              # Set to True to print face detection debugging info

# Portrait-specific cropping settings
PORTRAIT_FACE_WEIGHT = float(os.getenv("PORTRAIT_FACE_WEIGHT", "0.7"))      # Weight for face position in portrait cropping (0.0-1.0)
PORTRAIT_UPPER_BIAS = float(os.getenv("PORTRAIT_UPPER_BIAS", "0.3"))       # Bias toward upper portion of portrait (0.0-1.0)

def setup_directories():
    """
    Initialize the directory structure for image processing.
    
    Creates the output directory if it doesn't exist. MediaPipe face detection
    is built-in and doesn't require external model files.
    
    Returns:
        tuple: A 2-tuple containing:
            - image_dir (str): Absolute path to the source image directory
            - output_dir (str): Absolute path to the output directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, IMAGE_DIRECTORY) if IMAGE_DIRECTORY != "." else script_dir
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    
    return image_dir, output_dir

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
    """
    image_extensions = ['jpg', 'jpeg', 'png']  # Removed GIF - causes issues with face detection
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(directory, f"*.{ext}"), recursive=False))
        image_files.extend(glob(os.path.join(directory, f"*.{ext.upper()}"), recursive=False))
    
    return image_files

def detect_people_yolo(image):
    """
    Detect people in an image using YOLO when face detection fails.
    
    YOLO person detection is used as a fallback when RetinaFace doesn't
    find faces. This ensures we can still crop intelligently around people
    even if their faces aren't clearly visible or detectable.
    
    Args:
        image: OpenCV image in BGR format
    
    Returns:
        list: List of person dictionaries containing:
            - bbox: (x, y, width, height) bounding box
            - confidence: Detection confidence score (0.0-1.0)
            - center: (x, y) center point of the person
            - area: Person area in pixels
    """
    try:
        h, w = image.shape[:2]
        image_area = w * h
        detected_people = []
        
        # Initialize YOLO model (downloads automatically on first use)
        model = YOLO('yolov8n.pt')  # Nano model for speed
        
        # Run YOLO detection
        results = model(image, classes=[0], verbose=False)  # Class 0 = person
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Filter by confidence
                    if confidence >= MIN_FACE_CONFIDENCE:
                        # Calculate dimensions
                        x, y = x1, y1
                        width = x2 - x1
                        height = y2 - y1
                        person_area = width * height
                        person_center = (x + width // 2, y + height // 2)
                        
                        # Filter by size (avoid tiny detections)
                        if person_area / image_area >= FACE_SIZE_THRESHOLD:
                            person_dict = {
                                'bbox': (x, y, width, height),
                                'confidence': confidence,
                                'center': person_center,
                                'area': person_area,
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'right': x2,
                                'bottom': y2,
                                'type': 'person'  # Mark as person detection
                            }
                            detected_people.append(person_dict)
                            
                            if FACE_DEBUG:
                                print(f"    Person detected: conf={confidence:.3f}, "
                                      f"bbox=({x},{y},{width},{height}), "
                                      f"area={person_area}px ({person_area/image_area*100:.1f}% of image)")
        
        return detected_people
        
    except Exception as e:
        print(f"Error in YOLO person detection: {e}")
        return []

def detect_faces_retinaface(image):
    """
    Detect faces in an image using RetinaFace's state-of-the-art face detection.
    
    RetinaFace provides superior face detection accuracy (91.4% AP on WIDER FACE)
    with excellent handling of profile views, poor lighting, partial occlusion,
    and multiple faces. Particularly effective for portrait photography.
    
    Args:
        image: OpenCV image in BGR format
    
    Returns:
        list: List of face dictionaries containing:
            - bbox: (x, y, width, height) bounding box
            - confidence: Detection confidence score (0.0-1.0)
            - landmarks: 5-point facial landmarks (eyes, nose, mouth)
            - center: (x, y) center point of the face
            - area: Face area in pixels
    """
    try:
        h, w = image.shape[:2]
        image_area = w * h
        detected_faces = []
        
        # Use RetinaFace for detection - it expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # RetinaFace detection with threshold
        faces = RetinaFace.detect_faces(rgb_image, threshold=MIN_FACE_CONFIDENCE)
        
        if faces:
            for face_key, face_data in faces.items():
                # Extract bounding box coordinates
                facial_area = face_data['facial_area']
                x, y, right, bottom = facial_area
                
                # Calculate width and height
                width = right - x
                height = bottom - y
                
                # Calculate face properties
                face_area = width * height
                face_center = (x + width // 2, y + height // 2)
                
                # Get confidence score
                confidence = face_data['score']
                
                # Filter out very small faces (likely false positives)
                if face_area / image_area >= FACE_SIZE_THRESHOLD:
                    face_dict = {
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'center': face_center,
                        'area': face_area,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'right': right,
                        'bottom': bottom,
                        'landmarks': face_data['landmarks']  # 5-point landmarks
                    }
                    detected_faces.append(face_dict)
                    
                    if FACE_DEBUG:
                        print(f"    Face detected: conf={confidence:.3f}, "
                              f"bbox=({x},{y},{width},{height}), "
                              f"area={face_area}px ({face_area/image_area*100:.1f}% of image)")
        
        return detected_faces
        
    except Exception as e:
        print(f"Error in RetinaFace detection: {e}")
        return []

def calculate_portrait_aware_crop_region(faces, image_width, image_height, target_height, is_portrait=True):
    """
    Calculate optimal crop region with portrait-aware composition rules.
    
    This advanced algorithm is specifically designed for portrait→landscape cropping.
    It uses composition rules (rule of thirds), face quality assessment, and
    portrait-specific weighting to ensure natural, aesthetically pleasing crops.
    
    Args:
        faces: List of face dictionaries from detect_faces_retinaface
        image_width: Width of the source image
        image_height: Height of the source image
        target_height: Target height for the cropped image
        is_portrait: True if source image is portrait orientation
    
    Returns:
        tuple: (y_min, y_max) crop boundaries, or None if no valid crop found
    """
    if not faces:
        return None
    
    # Portrait-specific composition analysis
    if is_portrait:
        # For portraits, faces are typically in upper 1/3 to 1/2 of image
        upper_third = image_height // 3
        composition_center = int(image_height * 0.4)  # 40% down from top
        
        # Apply portrait-specific face weighting
        weighted_faces = []
        for face in faces:
            # Face quality assessment using landmarks (if available)
            face_quality = 1.0
            if 'landmarks' in face:
                # Assess face completeness and clarity
                landmarks = face['landmarks']
                if len(landmarks) >= 5:
                    # Calculate face symmetry and completeness
                    eye_distance = abs(landmarks['left_eye'][0] - landmarks['right_eye'][0])
                    face_quality = min(1.0, eye_distance / (face['width'] * 0.3))
            
            # Weight by face size (larger faces are more important)
            size_weight = face['area'] / (image_width * image_height)
            
            # Weight by position - favor upper portion for portraits
            y_position_ratio = face['center'][1] / image_height
            if y_position_ratio <= 0.5:  # Upper half gets bonus
                position_weight = 1.0 - (y_position_ratio * PORTRAIT_UPPER_BIAS)
            else:
                position_weight = 0.5 - ((y_position_ratio - 0.5) * 0.8)
            
            # Weight by proximity to portrait composition center
            center_distance = abs(face['center'][1] - composition_center) / image_height
            composition_weight = 1.0 - center_distance
            
            # Combined weight with portrait-specific emphasis
            total_weight = (size_weight * 2.0 + 
                          position_weight * PORTRAIT_FACE_WEIGHT + 
                          composition_weight * 1.5 + 
                          face_quality * 0.5)
            
            weighted_faces.append({
                'face': face,
                'weight': total_weight,
                'quality': face_quality
            })
    else:
        # Landscape images use traditional center-weighted approach
        weighted_faces = []
        image_center_y = image_height // 2
        
        for face in faces:
            size_weight = face['area'] / (image_width * image_height)
            center_distance = abs(face['center'][1] - image_center_y) / image_height
            centrality_weight = 1.0 - center_distance
            total_weight = size_weight * 2 + centrality_weight
            
            weighted_faces.append({
                'face': face,
                'weight': total_weight,
                'quality': 1.0
            })
    
    # Sort by weight (most important faces first)
    weighted_faces.sort(key=lambda x: x['weight'], reverse=True)
    
    # Calculate weighted center of mass with portrait bias
    total_weight = sum(wf['weight'] for wf in weighted_faces)
    weighted_center_y = sum(
        wf['face']['center'][1] * wf['weight'] for wf in weighted_faces
    ) / total_weight
    
    # Select primary faces (top 3 most important)
    primary_faces = [wf['face'] for wf in weighted_faces[:3]]
    
    # Find bounds that include all primary faces
    y_min = min(face['y'] for face in primary_faces)
    y_max = max(face['bottom'] for face in primary_faces)
    
    # Calculate adaptive padding based on face characteristics
    avg_face_height = sum(face['height'] for face in primary_faces) / len(primary_faces)
    base_padding = min(MAX_FACE_PADDING_PX, int(avg_face_height * FACE_PADDING_RATIO))
    
    # Apply portrait-specific padding adjustments
    if is_portrait:
        # More padding below faces in portraits (for shoulders/body)
        top_padding = int(base_padding * 0.8)
        bottom_padding = int(base_padding * 1.2)
    else:
        # Equal padding for landscape
        top_padding = bottom_padding = base_padding
    
    y_min = max(0, y_min - top_padding)
    y_max = min(image_height, y_max + bottom_padding)
    
    # Adjust to target height using composition-aware centering
    current_height = y_max - y_min
    
    if current_height < target_height:
        # Expand region with portrait-aware centering
        if is_portrait:
            # For portraits, bias toward upper portion
            expansion_needed = target_height - current_height
            upper_expansion = int(expansion_needed * 0.4)
            lower_expansion = expansion_needed - upper_expansion
            
            y_min = max(0, y_min - upper_expansion)
            y_max = min(image_height, y_max + lower_expansion)
            
            # Adjust if we hit bounds
            if y_max - y_min < target_height:
                if y_max == image_height:
                    y_min = max(0, y_max - target_height)
                elif y_min == 0:
                    y_max = min(image_height, y_min + target_height)
        else:
            # Standard center expansion for landscape
            y_min = max(0, int(weighted_center_y - target_height // 2))
            y_max = y_min + target_height
            
            if y_max > image_height:
                y_max = image_height
                y_min = max(0, y_max - target_height)
    
    elif current_height > target_height:
        # Shrink region with portrait-aware centering
        if is_portrait:
            # For portraits, prefer keeping upper portion
            y_max = min(image_height, y_min + target_height)
            if y_max == image_height:
                y_min = max(0, y_max - target_height)
        else:
            # Standard center shrinking for landscape
            y_min = max(0, int(weighted_center_y - target_height // 2))
            y_max = y_min + target_height
            
            if y_max > image_height:
                y_max = image_height
                y_min = max(0, y_max - target_height)
    
    # CRITICAL: Validate and fix face boundaries to prevent cutoffs
    face_y_min = min(face['y'] for face in primary_faces)
    face_y_max = max(face['bottom'] for face in primary_faces)
    
    # Check if any faces would be cut off
    faces_in_bounds = (face_y_min >= y_min and face_y_max <= y_max)
    
    if not faces_in_bounds:
        if FACE_DEBUG:
            print(f"    FIXING: Faces would be cut off! Adjusting crop bounds...")
            print(f"    Original crop: Y={y_min}-{y_max}, Face bounds: Y={face_y_min}-{face_y_max}")
        
        # Ensure all faces fit within target height
        required_height = face_y_max - face_y_min
        if required_height <= target_height:
            # Add generous padding around faces
            face_padding = min(MAX_FACE_PADDING_PX, int((target_height - required_height) * 0.4))
            
            # Start with face bounds plus padding
            y_min = max(0, face_y_min - face_padding)
            y_max = min(image_height, face_y_max + face_padding)
            
            # Adjust to exact target height
            current_height = y_max - y_min
            if current_height < target_height:
                # Need to expand
                expansion_needed = target_height - current_height
                if is_portrait:
                    # For portraits, expand more downward (for body)
                    expand_up = min(expansion_needed // 3, y_min)
                    expand_down = expansion_needed - expand_up
                    y_min = max(0, y_min - expand_up)
                    y_max = min(image_height, y_max + expand_down)
                else:
                    # For landscape, expand equally
                    expand_each = expansion_needed // 2
                    y_min = max(0, y_min - expand_each)
                    y_max = min(image_height, y_max + expand_each)
            
            # Final adjustment if we hit image bounds
            if y_max - y_min < target_height:
                if y_max == image_height:
                    y_min = max(0, y_max - target_height)
                elif y_min == 0:
                    y_max = min(image_height, y_min + target_height)
            
            if FACE_DEBUG:
                print(f"    FIXED crop: Y={y_min}-{y_max} (height={y_max-y_min})")
        else:
            if FACE_DEBUG:
                print(f"    WARNING: Faces too tall ({required_height}px) for target height ({target_height}px)")
            # Faces are too tall - crop from top of highest face
            y_min = face_y_min
            y_max = min(image_height, y_min + target_height)
    
    # Final validation
    final_faces_in_bounds = all(
        face['y'] >= y_min and face['bottom'] <= y_max
        for face in primary_faces
    )
    
    if not final_faces_in_bounds and FACE_DEBUG:
        print(f"    ERROR: Still cutting off faces after adjustment!")
        for i, face in enumerate(primary_faces):
            in_bounds = face['y'] >= y_min and face['bottom'] <= y_max
            print(f"      Face {i+1}: Y={face['y']}-{face['bottom']}, In bounds: {in_bounds}")
    
    return (y_min, y_max)

def process_image(args):
    """
    Process a single image with intelligent cropping and resizing.
    
    This function handles the core image processing pipeline:
    1. Load and validate the image
    2. Determine orientation (landscape vs portrait)
    3. Resize to fit target dimensions while maintaining aspect ratio
    4. Apply intelligent cropping using face detection when available
    5. Save the processed image to the output directory
    
    The cropping algorithm uses advanced MediaPipe face detection with smart
    prioritization to ensure faces remain visible and well-composed in the final image.
    When faces are detected, the crop boundaries are calculated using weighted
    center-of-mass and face importance scoring. If no faces are detected,
    the image is cropped from the center.
    
    Args:
        args (tuple): A 2-tuple containing:
            - image_file (str): Path to the source image file
            - output_dir (str): Directory where processed image will be saved
    
    Returns:
        str or None: Filename of the successfully processed image, or None if processing failed
        
    Raises:
        Exception: Various exceptions may occur during image processing, which are caught
                  and logged with the specific image filename for debugging
    """
    image_file, output_dir = args
    
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
            # Use INTER_LANCZOS4 for better quality resize
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # If resized image is taller than screen, we need to crop vertically
            if new_h > target_height:
                # Try RetinaFace first for face detection
                faces = detect_faces_retinaface(resized_image)
                
                if faces:
                    if FACE_DEBUG:
                        print(f"Found {len(faces)} faces in {os.path.basename(image_file)} (landscape)")
                    
                    # Calculate optimal crop region using portrait-aware algorithm
                    crop_region = calculate_portrait_aware_crop_region(faces, new_w, new_h, target_height, is_portrait=False)
                    
                    if crop_region:
                        y_min, y_max = crop_region
                        
                        if FACE_DEBUG:
                            print(f"  Face-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                        
                        cropped_image = resized_image[y_min:y_max, :]
                    else:
                        # Face detection found faces but crop failed - try person detection
                        people = detect_people_yolo(resized_image)
                        if people:
                            if FACE_DEBUG:
                                print(f"  Face crop failed, found {len(people)} people (landscape)")
                            crop_region = calculate_portrait_aware_crop_region(people, new_w, new_h, target_height, is_portrait=False)
                            if crop_region:
                                y_min, y_max = crop_region
                                if FACE_DEBUG:
                                    print(f"  Person-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                                cropped_image = resized_image[y_min:y_max, :]
                            else:
                                # Both failed - center crop
                                y_center = new_h // 2
                                y_min = max(0, y_center - target_height//2)
                                y_max = min(new_h, y_center + target_height//2)
                                cropped_image = resized_image[y_min:y_max, :]
                        else:
                            # Center crop fallback
                            y_center = new_h // 2
                            y_min = max(0, y_center - target_height//2)
                            y_max = min(new_h, y_center + target_height//2)
                            cropped_image = resized_image[y_min:y_max, :]
                else:
                    # No faces detected - try person detection
                    people = detect_people_yolo(resized_image)
                    if people:
                        if FACE_DEBUG:
                            print(f"No faces found, detected {len(people)} people in {os.path.basename(image_file)} (landscape)")
                        
                        crop_region = calculate_portrait_aware_crop_region(people, new_w, new_h, target_height, is_portrait=False)
                        if crop_region:
                            y_min, y_max = crop_region
                            if FACE_DEBUG:
                                print(f"  Person-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                            cropped_image = resized_image[y_min:y_max, :]
                        else:
                            # Person detection failed - center crop
                            y_center = new_h // 2
                            y_min = max(0, y_center - target_height//2)
                            y_max = min(new_h, y_center + target_height//2)
                            cropped_image = resized_image[y_min:y_max, :]
                    else:
                        # No people detected - use center cropping
                        if FACE_DEBUG:
                            print(f"No faces or people detected in {os.path.basename(image_file)} (landscape) - using center crop")
                        y_center = new_h // 2
                        y_min = max(0, y_center - target_height//2)
                        y_max = min(new_h, y_center + target_height//2)
                        cropped_image = resized_image[y_min:y_max, :]
                
                final_image = cropped_image
            else:
                final_image = resized_image
        
        # Process portrait images (taller than wide) - This is the main use case!
        else:
            # Scale image to fit screen width, maintaining aspect ratio
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            # Use INTER_LANCZOS4 for better quality resize
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Try RetinaFace first for intelligent portrait cropping
            faces = detect_faces_retinaface(resized_image)
            
            if faces:
                if FACE_DEBUG:
                    print(f"Found {len(faces)} faces in {os.path.basename(image_file)} (portrait)")
                
                # Calculate optimal crop region using portrait-aware algorithm
                crop_region = calculate_portrait_aware_crop_region(faces, new_w, new_h, target_height, is_portrait=True)
                
                if crop_region:
                    y_min, y_max = crop_region
                    
                    if FACE_DEBUG:
                        print(f"  Face-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                    
                    cropped_image = resized_image[y_min:y_max, :]
                    final_image = cropped_image
                else:
                    # Face detection found faces but crop failed - try person detection
                    people = detect_people_yolo(resized_image)
                    if people:
                        if FACE_DEBUG:
                            print(f"  Face crop failed, found {len(people)} people (portrait)")
                        crop_region = calculate_portrait_aware_crop_region(people, new_w, new_h, target_height, is_portrait=True)
                        if crop_region:
                            y_min, y_max = crop_region
                            if FACE_DEBUG:
                                print(f"  Person-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                            cropped_image = resized_image[y_min:y_max, :]
                            final_image = cropped_image
                        else:
                            # Both failed - center crop
                            y_center = new_h // 2
                            y_min = max(0, y_center - target_height//2)
                            y_max = min(new_h, y_center + target_height//2)
                            cropped_image = resized_image[y_min:y_max, :]
                            final_image = cropped_image
                    else:
                        # Center crop fallback
                        y_center = new_h // 2
                        y_min = max(0, y_center - target_height//2)
                        y_max = min(new_h, y_center + target_height//2)
                        cropped_image = resized_image[y_min:y_max, :]
                        final_image = cropped_image
            else:
                # No faces detected - try person detection as fallback
                people = detect_people_yolo(resized_image)
                if people:
                    if FACE_DEBUG:
                        print(f"No faces found, detected {len(people)} people in {os.path.basename(image_file)} (portrait)")
                    
                    crop_region = calculate_portrait_aware_crop_region(people, new_w, new_h, target_height, is_portrait=True)
                    if crop_region:
                        y_min, y_max = crop_region
                        if FACE_DEBUG:
                            print(f"  Person-based crop region: Y={y_min}-{y_max} (height={y_max-y_min})")
                        cropped_image = resized_image[y_min:y_max, :]
                        final_image = cropped_image
                    else:
                        # Person detection failed - center crop
                        y_center = new_h // 2
                        y_min = max(0, y_center - target_height//2)
                        y_max = min(new_h, y_center + target_height//2)
                        cropped_image = resized_image[y_min:y_max, :]
                        final_image = cropped_image
                else:
                    # No people detected - use center cropping
                    if FACE_DEBUG:
                        print(f"No faces or people detected in {os.path.basename(image_file)} (portrait) - using center crop")
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

    #clock {{
        position: fixed;
        bottom: 30px;
        left: 30px;
        font-family: Arial, sans-serif;
        font-size: 64px;
        font-weight: bold;
        color: white;
        z-index: 100;
        text-shadow: 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000, -2px 0 0 #000;
    }}

    #weather {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        font-family: Arial, sans-serif;
        font-size: 64px;
        font-weight: bold;
        color: white;
        z-index: 100;
        text-shadow: 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000, -2px 0 0 #000;
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
  <div id="clock"></div>
  <div id="weather"></div>

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
      
      // Check if API key is configured
      if (apiKey === 'YOUR_API_KEY_HERE' || !apiKey || apiKey.length < 10) {{
        weatherDiv.innerHTML = "Configure API key";
        console.error('Weather API key not configured. Edit WEATHER_API_KEY in the script.');
        return;
      }}
      
      var url = `https://api.openweathermap.org/data/2.5/weather?zip=${{zipCode}}&units=imperial&appid=${{apiKey}}`;
      console.log('Weather API URL (without key):', url.replace(apiKey, '[HIDDEN]'));

      var currentTime = new Date().getTime();
      // Only update weather every 5 minutes (300000ms) to avoid rate limiting
      if (currentTime - lastWeatherUpdate >= 300000) {{
        console.log('Fetching weather data...');
        weatherDiv.innerHTML = "Loading weather...";
        
        fetch(url)
          .then(response => {{
            console.log('Weather API response status:', response.status);
            if (!response.ok) {{
              throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
            }}
            return response.json();
          }})
          .then(data => {{
            console.log('Weather data received:', data);
            // Check if response has expected structure
            if (!data.main || !data.weather || !data.weather[0]) {{
              throw new Error('Invalid weather data structure');
            }}
            // Extract temperature and weather icon from API response
            var temperature = Math.round(data.main.temp);
            var icon = data.weather[0].icon;
            var iconUrl = `https://openweathermap.org/img/wn/${{icon}}@2x.png`;
            // Display weather icon and temperature
            weatherDiv.innerHTML = `<img src="${{iconUrl}}" alt="weather icon"> ${{temperature}}°F`;
            lastWeatherUpdate = currentTime;
            console.log('Weather updated successfully');
          }})
          .catch(error => {{
            // Handle API errors gracefully with detailed error info
            console.error('Weather API error:', error);
            var errorMsg = "Weather unavailable";
            if (error.message.includes('401')) {{
              errorMsg = "Invalid API key";
            }} else if (error.message.includes('404')) {{
              errorMsg = "Invalid ZIP code";
            }} else if (error.message.includes('429')) {{
              errorMsg = "Rate limit exceeded";
            }}
            weatherDiv.innerHTML = errorMsg;
          }});
      }} else {{
        console.log('Weather update skipped (too soon)');
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
      // Check for weather updates every 5 minutes (300000ms)
      setInterval(function() {{
        updateWeather(zipCode);
      }}, 300000);
      
      console.log('Slideshow initialized with ZIP code:', zipCode);
      
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
    1. Set up directories
    2. Discover all image files in the source directory
    3. Process all images in parallel using multiprocessing with MediaPipe face detection
    4. Generate the HTML slideshow with processed images
    5. Provide user feedback and final instructions
    
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
    image_dir, output_dir = setup_directories()
    
    print(f"Looking for images in: {image_dir}")
    image_files = find_image_files(image_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # RetinaFace + YOLO provides comprehensive subject detection
    print("RetinaFace face detection + YOLO person detection enabled")
    
    # Prepare argument tuples for parallel processing
    # Each worker process needs: (image_path, output_dir)
    process_args = [(img_file, output_dir) for img_file in image_files]
    
    print("Processing images with RetinaFace + YOLO detection and portrait-aware cropping...")
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