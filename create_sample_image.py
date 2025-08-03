#!/usr/bin/env python3
"""
Create a sample image with a person for testing the video generator
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_person_image():
    """Create a sample image with a person figure"""
    # Create canvas
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add gradient background
    for i in range(height):
        intensity = int(150 + 50 * (i / height))
        img[i, :] = [intensity - 20, intensity, intensity + 20]
    
    # Draw a simple person figure
    person_x = width // 2
    person_y = height // 2
    
    # Head (circle)
    cv2.circle(img, (person_x, person_y - 100), 40, (220, 180, 140), -1)  # Skin tone
    cv2.circle(img, (person_x, person_y - 100), 40, (0, 0, 0), 2)  # Outline
    
    # Eyes
    cv2.circle(img, (person_x - 15, person_y - 110), 5, (0, 0, 0), -1)
    cv2.circle(img, (person_x + 15, person_y - 110), 5, (0, 0, 0), -1)
    
    # Smile
    cv2.ellipse(img, (person_x, person_y - 90), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Body (rectangle)
    cv2.rectangle(img, (person_x - 30, person_y - 60), (person_x + 30, person_y + 80), (100, 150, 200), -1)
    cv2.rectangle(img, (person_x - 30, person_y - 60), (person_x + 30, person_y + 80), (0, 0, 0), 2)
    
    # Arms
    cv2.line(img, (person_x - 30, person_y - 40), (person_x - 70, person_y), (220, 180, 140), 15)
    cv2.line(img, (person_x + 30, person_y - 40), (person_x + 70, person_y), (220, 180, 140), 15)
    
    # Hands (circles)
    cv2.circle(img, (person_x - 70, person_y), 12, (220, 180, 140), -1)
    cv2.circle(img, (person_x + 70, person_y), 12, (220, 180, 140), -1)
    
    # Legs
    cv2.line(img, (person_x - 15, person_y + 80), (person_x - 15, person_y + 150), (100, 150, 200), 20)
    cv2.line(img, (person_x + 15, person_y + 80), (person_x + 15, person_y + 150), (100, 150, 200), 20)
    
    # Feet
    cv2.ellipse(img, (person_x - 15, person_y + 160), (25, 10), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (person_x + 15, person_y + 160), (25, 10), 0, 0, 360, (50, 50, 50), -1)
    
    # Add some decorative elements
    # Sun
    cv2.circle(img, (width - 100, 100), 30, (255, 255, 0), -1)
    for i in range(8):
        angle = i * 45 * np.pi / 180
        x1 = int(width - 100 + 45 * np.cos(angle))
        y1 = int(100 + 45 * np.sin(angle))
        x2 = int(width - 100 + 60 * np.cos(angle))
        y2 = int(100 + 60 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
    
    # Ground line
    cv2.line(img, (0, person_y + 170), (width, person_y + 170), (100, 200, 100), 5)
    
    return img

def main():
    """Create sample image and save to folder X"""
    os.makedirs("X", exist_ok=True)
    
    # Create sample image
    img = create_sample_person_image()
    
    # Save image
    output_path = "X/sample_person.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Sample image created: {output_path}")
    print("You can now run the video generator!")

if __name__ == "__main__":
    main()