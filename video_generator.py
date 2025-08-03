#!/usr/bin/env python3
"""
AI-Powered Vertical Video Generator
Generates viral-style vertical videos with animated characters and effects
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from moviepy.editor import *
import random
import math
from typing import Tuple, List
import json
import time

class VerticalVideoGenerator:
    def __init__(self):
        self.width = 1080
        self.height = 1920
        self.fps = 30
        self.duration_range = (15, 30)
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and prepare the input image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def resize_and_center_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to fit vertical format while maintaining aspect ratio"""
        h, w = img.shape[:2]
        
        # Calculate scaling to fit within vertical frame
        scale_w = self.width / w
        scale_h = self.height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create vertical canvas and center the image
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(self.height):
            intensity = int(20 + 30 * (i / self.height))
            canvas[i, :] = [intensity, intensity + 10, intensity + 20]
        
        # Center the image
        y_offset = (self.height - new_h) // 2
        x_offset = (self.width - new_w) // 2
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        
        return canvas, (x_offset, y_offset, new_w, new_h)
    
    def create_person_mask(self, img: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a rough mask for the person in the image"""
        x, y, w, h = person_bbox
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Create elliptical mask for person
        center_x = x + w // 2
        center_y = y + h // 2
        
        cv2.ellipse(mask, (center_x, center_y), (w//2, h//2), 0, 0, 360, 255, -1)
        
        return mask
    
    def animate_person(self, base_img: np.ndarray, person_bbox: Tuple[int, int, int, int], frame_num: int, total_frames: int) -> np.ndarray:
        """Animate the person with walking motion and happiness gestures"""
        img = base_img.copy()
        x, y, w, h = person_bbox
        
        # Walking animation - subtle horizontal movement
        walk_cycle = math.sin(frame_num * 0.3) * 3
        
        # Happy gestures - arm raising animation
        progress = frame_num / total_frames
        arm_raise = math.sin(progress * math.pi * 2) * 10
        
        # Create transformation matrix for subtle animation
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Apply subtle transformations
        M = cv2.getRotationMatrix2D((center_x, center_y), math.sin(frame_num * 0.1) * 2, 1.0)
        M[0, 2] += walk_cycle
        M[1, 2] += arm_raise * 0.3
        
        # Apply transformation to person region only
        person_region = img[y:y+h, x:x+w]
        if person_region.size > 0:
            transformed = cv2.warpAffine(person_region, M[:, :2], (w, h))
            img[y:y+h, x:x+w] = transformed
        
        return img
    
    def create_dollar_bill(self, size: int = 60) -> np.ndarray:
        """Create a dollar bill image"""
        bill = np.zeros((size, size*2, 3), dtype=np.uint8)
        bill[:] = [34, 139, 34]  # Dark green
        
        # Add dollar sign
        cv2.putText(bill, '$', (size//3, size//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.rectangle(bill, (5, 5), (size*2-5, size-5), (255, 255, 255), 2)
        
        return bill
    
    def add_falling_dollars(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        """Add animated falling dollar bills"""
        result = img.copy()
        dollar_bill = self.create_dollar_bill()
        
        # Create multiple dollar bills falling
        num_bills = 8
        for i in range(num_bills):
            # Calculate position based on frame and bill index
            x = (i * 150 + frame_num * 2) % (self.width + 100) - 50
            y = (frame_num * 5 + i * 200) % (self.height + 200) - 100
            
            if x > -100 and x < self.width and y > -100 and y < self.height:
                # Add rotation
                rotation = (frame_num + i * 30) % 360
                rows, cols = dollar_bill.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
                rotated_bill = cv2.warpAffine(dollar_bill, M, (cols, rows))
                
                # Blend onto main image
                h, w = rotated_bill.shape[:2]
                if x >= 0 and y >= 0 and x + w < self.width and y + h < self.height:
                    # Create alpha blend
                    alpha = 0.8
                    result[y:y+h, x:x+w] = cv2.addWeighted(
                        result[y:y+h, x:x+w], 1-alpha, rotated_bill, alpha, 0
                    )
        
        return result
    
    def add_sparkle_effects(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        """Add sparkle and glow effects"""
        result = img.copy()
        
        # Add sparkles
        num_sparkles = 15
        for i in range(num_sparkles):
            x = (frame_num * 3 + i * 123) % self.width
            y = (frame_num * 2 + i * 456) % self.height
            
            # Sparkle intensity based on sine wave
            intensity = int(128 + 127 * math.sin(frame_num * 0.1 + i))
            
            # Draw sparkle
            cv2.circle(result, (x, y), 3, (255, 255, intensity), -1)
            cv2.circle(result, (x, y), 6, (255, 255, intensity//2), 1)
        
        # Add glow effect around person area (assuming center region)
        glow_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.ellipse(glow_mask, (self.width//2, self.height//2), 
                   (200, 300), 0, 0, 360, 100, -1)
        
        # Apply Gaussian blur for glow
        glow_mask = cv2.GaussianBlur(glow_mask, (51, 51), 0)
        
        # Add golden glow
        glow_color = np.array([255, 215, 0], dtype=np.uint8)
        for c in range(3):
            result[:, :, c] = cv2.addWeighted(
                result[:, :, c], 1.0, 
                (glow_mask * glow_color[c] // 255).astype(np.uint8), 0.3, 0
            )
        
        return result
    
    def add_motivational_text(self, img: np.ndarray, frame_num: int, total_frames: int) -> np.ndarray:
        """Add motivational text overlay"""
        result = img.copy()
        
        # Text appears in different phases
        progress = frame_num / total_frames
        
        texts = [
            "MONEY FLOWS TO ME",
            "ABUNDANCE MINDSET",
            "SUCCESS IS COMING"
        ]
        
        if progress < 0.3:
            text = texts[0]
        elif progress < 0.7:
            text = texts[1]
        else:
            text = texts[2]
        
        # Text animation - fade in/out and scale
        phase_progress = (progress * 3) % 1
        alpha = min(phase_progress * 2, (1 - phase_progress) * 2, 1.0)
        
        # Create text image
        font_scale = 2.0 + math.sin(frame_num * 0.1) * 0.3
        thickness = 4
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)
        
        # Position text
        x = (self.width - text_width) // 2
        y = 150
        
        # Add text with outline
        cv2.putText(result, text, (x-2, y-2), cv2.FONT_HERSHEY_BOLD, font_scale, (0, 0, 0), thickness+2)
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_BOLD, font_scale, (255, 255, 0), thickness)
        
        return result
    
    def create_background_music(self, duration: float, output_path: str):
        """Create motivational background music using simple tones"""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create uplifting chord progression
        frequencies = [261.63, 329.63, 392.00, 523.25]  # C, E, G, C
        music = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            phase_offset = i * np.pi / 4
            wave = 0.3 * np.sin(2 * np.pi * freq * t + phase_offset)
            # Add some rhythm
            envelope = np.abs(np.sin(2 * np.pi * t * 2))
            music += wave * envelope
        
        # Add bass line
        bass_freq = 130.81  # C2
        bass = 0.4 * np.sin(2 * np.pi * bass_freq * t) * np.abs(np.sin(2 * np.pi * t))
        music += bass
        
        # Normalize
        music = music / np.max(np.abs(music)) * 0.7
        
        # Convert to 16-bit
        music_16bit = (music * 32767).astype(np.int16)
        
        # Save as WAV (moviepy can handle this)
        import wave
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(music_16bit.tobytes())
    
    def generate_video(self, image_path: str, output_path: str, prompt: str = None):
        """Generate the complete vertical video"""
        print(f"Loading image from: {image_path}")
        
        # Load and prepare image
        base_img = self.load_image(image_path)
        canvas, person_bbox = self.resize_and_center_image(base_img)
        
        # Determine video duration
        duration = random.uniform(*self.duration_range)
        total_frames = int(duration * self.fps)
        
        print(f"Generating video: {duration:.1f}s, {total_frames} frames")
        
        # Generate frames
        frames = []
        for frame_num in range(total_frames):
            if frame_num % 30 == 0:
                print(f"Processing frame {frame_num}/{total_frames}")
            
            # Start with base canvas
            frame = canvas.copy()
            
            # Animate person
            frame = self.animate_person(frame, person_bbox, frame_num, total_frames)
            
            # Add falling dollars
            frame = self.add_falling_dollars(frame, frame_num)
            
            # Add sparkle effects
            frame = self.add_sparkle_effects(frame, frame_num)
            
            # Add motivational text
            frame = self.add_motivational_text(frame, frame_num, total_frames)
            
            frames.append(frame)
        
        # Create video clip
        print("Creating video clip...")
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames], fps=self.fps)
        
        # Create background music
        music_path = "temp_music.wav"
        print("Generating background music...")
        self.create_background_music(duration, music_path)
        
        # Add audio
        audio = AudioFileClip(music_path)
        final_clip = clip.set_audio(audio)
        
        # Write video
        print(f"Saving video to: {output_path}")
        final_clip.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            preset='medium',
            ffmpeg_params=['-crf', '23']
        )
        
        # Cleanup
        if os.path.exists(music_path):
            os.remove(music_path)
        
        print(f"Video generated successfully: {output_path}")

def main():
    """Main function to process images from folder X and output to folder Y"""
    generator = VerticalVideoGenerator()
    
    input_folder = "X"
    output_folder = "Y"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all images in folder X
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_animated.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"\n--- Processing: {filename} ---")
            
            try:
                generator.generate_video(input_path, output_path)
                print(f"✅ Successfully generated: {output_filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
    
    print("\n🎉 All videos generated and saved to folder Y!")

if __name__ == "__main__":
    main()