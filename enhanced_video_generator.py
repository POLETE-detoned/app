#!/usr/bin/env python3
"""
Enhanced AI-Powered Vertical Video Generator
Processes images from folder X and creates animated vertical videos
"""

import os
import cv2
import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageEnhance
import time

class EnhancedVideoGenerator:
    def __init__(self):
        self.width = 1080
        self.height = 1920
        self.fps = 30
        self.duration_range = (15, 30)
    
    def load_and_process_image(self, image_path):
        """Load and process input image"""
        print(f"Loading image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and center image for vertical format
        h, w = img.shape[:2]
        
        # Calculate scaling to fit within vertical frame
        scale_w = self.width / w
        scale_h = self.height / h
        scale = min(scale_w, scale_h) * 0.8  # Leave some margin
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create vertical canvas with gradient background
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(self.height):
            intensity = int(20 + 40 * (i / self.height))
            canvas[i, :] = [intensity, intensity + 10, intensity + 25]
        
        # Center the image
        y_offset = (self.height - new_h) // 2
        x_offset = (self.width - new_w) // 2
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        
        return canvas, (x_offset, y_offset, new_w, new_h)
    
    def create_dollar_bill(self, size=50):
        """Create a dollar bill graphic"""
        bill = np.zeros((size, size*2, 3), dtype=np.uint8)
        bill[:] = [34, 139, 34]  # Dark green
        
        # Add border
        cv2.rectangle(bill, (2, 2), (size*2-2, size-2), (255, 255, 255), 2)
        
        # Add dollar sign
        cv2.putText(bill, '$', (size//2, size//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return bill
    
    def add_falling_money(self, frame, frame_num):
        """Add animated falling money"""
        result = frame.copy()
        dollar_bill = self.create_dollar_bill()
        
        # Multiple dollar bills falling
        num_bills = 12
        for i in range(num_bills):
            # Calculate position with different speeds
            x = (i * 120 + frame_num * (2 + i % 3)) % (self.width + 150) - 75
            y = (frame_num * (4 + i % 2) + i * 180) % (self.height + 250) - 125
            
            if x > -100 and x < self.width and y > -100 and y < self.height:
                # Add rotation for more dynamic effect
                angle = (frame_num * 2 + i * 45) % 360
                
                # Rotate dollar bill
                h, w = dollar_bill.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(dollar_bill, M, (w, h))
                
                # Blend onto frame
                end_x = min(x + w, self.width)
                end_y = min(y + h, self.height)
                start_x = max(x, 0)
                start_y = max(y, 0)
                
                if start_x < end_x and start_y < end_y:
                    bill_start_x = max(0, -x)
                    bill_start_y = max(0, -y)
                    bill_end_x = bill_start_x + (end_x - start_x)
                    bill_end_y = bill_start_y + (end_y - start_y)
                    
                    # Alpha blending
                    alpha = 0.7
                    result[start_y:end_y, start_x:end_x] = cv2.addWeighted(
                        result[start_y:end_y, start_x:end_x], 1-alpha,
                        rotated[bill_start_y:bill_end_y, bill_start_x:bill_end_x], alpha, 0
                    )
        
        return result
    
    def add_sparkle_effects(self, frame, frame_num):
        """Add sparkle and glow effects"""
        result = frame.copy()
        
        # Add sparkles with different sizes and intensities
        num_sparkles = 20
        for i in range(num_sparkles):
            x = (frame_num * 3 + i * 137) % self.width
            y = (frame_num * 2 + i * 211) % self.height
            
            # Sparkle intensity with sine wave
            intensity = int(100 + 155 * abs(math.sin(frame_num * 0.1 + i)))
            size = 2 + int(3 * abs(math.sin(frame_num * 0.05 + i)))
            
            # Main sparkle
            cv2.circle(result, (x, y), size, (255, 255, intensity), -1)
            
            # Outer glow
            cv2.circle(result, (x, y), size + 3, (255, 255, intensity//3), 1)
            
            # Cross pattern for some sparkles
            if i % 3 == 0:
                cv2.line(result, (x-8, y), (x+8, y), (255, 255, intensity//2), 1)
                cv2.line(result, (x, y-8), (x, y+8), (255, 255, intensity//2), 1)
        
        return result
    
    def add_motivational_text(self, frame, frame_num, total_frames):
        """Add animated motivational text"""
        result = frame.copy()
        
        # Text rotation through video
        progress = frame_num / total_frames
        
        texts = [
            "MONEY FLOWS TO ME",
            "ABUNDANCE MINDSET", 
            "SUCCESS IS MINE",
            "WEALTH ATTRACTION",
            "PROSPERITY NOW"
        ]
        
        # Select text based on progress
        text_index = int(progress * len(texts))
        if text_index >= len(texts):
            text_index = len(texts) - 1
        text = texts[text_index]
        
        # Text animation effects
        scale_wave = 1.8 + 0.4 * math.sin(frame_num * 0.12)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 4
        
        # Get text dimensions
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale_wave, thickness)
        
        # Position text at top
        x = (self.width - text_width) // 2
        y = 120 + int(10 * math.sin(frame_num * 0.08))  # Slight vertical bounce
        
        # Add multiple outline layers for better visibility
        outline_colors = [(0, 0, 0), (50, 50, 50)]
        outline_thicknesses = [thickness + 4, thickness + 2]
        
        for color, thick in zip(outline_colors, outline_thicknesses):
            cv2.putText(result, text, (x, y), font, scale_wave, color, thick)
        
        # Main text with golden color
        cv2.putText(result, text, (x, y), font, scale_wave, (0, 215, 255), thickness)
        
        return result
    
    def add_dynamic_effects(self, frame, frame_num):
        """Add dynamic visual effects"""
        result = frame.copy()
        
        # Add pulsing glow around the edges
        glow_intensity = int(30 + 25 * abs(math.sin(frame_num * 0.1)))
        
        # Top and bottom glow bars
        cv2.rectangle(result, (0, 0), (self.width, 50), (glow_intensity, glow_intensity*2, glow_intensity*3), -1)
        cv2.rectangle(result, (0, self.height-50), (self.width, self.height), (glow_intensity, glow_intensity*2, glow_intensity*3), -1)
        
        # Side glows
        cv2.rectangle(result, (0, 0), (30, self.height), (glow_intensity//2, glow_intensity, glow_intensity*2), -1)
        cv2.rectangle(result, (self.width-30, 0), (self.width, self.height), (glow_intensity//2, glow_intensity, glow_intensity*2), -1)
        
        return result
    
    def generate_frames(self, base_canvas, person_bbox, duration):
        """Generate all video frames"""
        total_frames = int(duration * self.fps)
        frames = []
        
        print(f"Generating {total_frames} frames for {duration:.1f}s video...")
        
        for frame_num in range(total_frames):
            if frame_num % 60 == 0:
                print(f"Processing frame {frame_num}/{total_frames}")
            
            # Start with base canvas
            frame = base_canvas.copy()
            
            # Add person animation (subtle movement)
            x, y, w, h = person_bbox
            
            # Subtle breathing/movement animation
            scale_factor = 1.0 + 0.02 * math.sin(frame_num * 0.15)
            offset_x = int(3 * math.sin(frame_num * 0.08))
            offset_y = int(2 * math.sin(frame_num * 0.12))
            
            # Apply subtle transformation to person region
            if w > 0 and h > 0:
                person_region = frame[y:y+h, x:x+w].copy()
                center_x, center_y = w//2, h//2
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale_factor)
                M[0, 2] += offset_x
                M[1, 2] += offset_y
                
                # Apply transformation
                transformed = cv2.warpAffine(person_region, M, (w, h))
                frame[y:y+h, x:x+w] = transformed
            
            # Add falling money
            frame = self.add_falling_money(frame, frame_num)
            
            # Add sparkle effects
            frame = self.add_sparkle_effects(frame, frame_num)
            
            # Add motivational text
            frame = self.add_motivational_text(frame, frame_num, total_frames)
            
            # Add dynamic effects
            frame = self.add_dynamic_effects(frame, frame_num)
            
            frames.append(frame)
        
        return frames
    
    def create_video(self, frames, fps, output_path):
        """Create video file from frames"""
        print(f"Creating video with {len(frames)} frames...")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(frames):
            if i % 60 == 0:
                print(f"Writing frame {i}/{len(frames)}")
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {output_path}")
    
    def generate_audio(self, duration, output_path):
        """Generate uplifting background music"""
        try:
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create uplifting music with multiple harmonies
            frequencies = [261.63, 329.63, 392.00, 523.25, 659.25]  # C, E, G, C, E
            music = np.zeros_like(t)
            
            for i, freq in enumerate(frequencies):
                phase = i * np.pi / 3
                amplitude = 0.15 / (i + 1)  # Decreasing amplitude for harmonics
                wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
                
                # Add rhythmic envelope
                envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * t * 1.5))
                music += wave * envelope
            
            # Add bass line
            bass_freq = 130.81  # C2
            bass = 0.25 * np.sin(2 * np.pi * bass_freq * t)
            bass_envelope = 0.7 + 0.3 * np.abs(np.sin(2 * np.pi * t * 0.75))
            music += bass * bass_envelope
            
            # Add some higher frequency sparkle
            sparkle_freq = 1046.50  # C6
            sparkle = 0.05 * np.sin(2 * np.pi * sparkle_freq * t)
            sparkle_envelope = np.abs(np.sin(2 * np.pi * t * 4))
            music += sparkle * sparkle_envelope
            
            # Normalize and convert
            music = music / np.max(np.abs(music)) * 0.8
            music_16bit = (music * 32767).astype(np.int16)
            
            # Save as WAV
            import wave
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(music_16bit.tobytes())
            
            print(f"Audio generated: {output_path}")
            return True
        except Exception as e:
            print(f"Audio generation failed: {e}")
            return False
    
    def combine_video_audio(self, video_path, audio_path, output_path):
        """Combine video and audio using ffmpeg"""
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-crf', '23',
                '-preset', 'medium',
                '-shortest',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Final video created: {output_path}")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
        except Exception as e:
            print(f"Video combination failed: {e}")
            return False
    
    def process_image(self, image_path, output_dir):
        """Process a single image into an animated video"""
        print(f"\n🎬 Processing: {os.path.basename(image_path)}")
        
        # Load and process image
        base_canvas, person_bbox = self.load_and_process_image(image_path)
        
        # Generate random duration within range
        duration = random.uniform(*self.duration_range)
        
        # Generate frames
        frames = self.generate_frames(base_canvas, person_bbox, duration)
        
        # Create temporary video
        temp_video = "temp_video.mp4"
        self.create_video(frames, self.fps, temp_video)
        
        # Generate audio
        temp_audio = "temp_audio.wav"
        audio_created = self.generate_audio(duration, temp_audio)
        
        # Create final output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        final_output = os.path.join(output_dir, f"{base_name}_animated.mp4")
        
        # Combine video and audio
        if audio_created and os.path.exists(temp_audio):
            if self.combine_video_audio(temp_video, temp_audio, final_output):
                # Clean up temp files
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            else:
                # If combining failed, use video without audio
                import shutil
                shutil.move(temp_video, final_output)
        else:
            # Use video without audio
            import shutil
            shutil.move(temp_video, final_output)
        
        print(f"✅ Generated: {final_output}")
        print(f"   Duration: {duration:.1f}s | Resolution: 1080x1920")
        
        return final_output

def main():
    """Main function to process all images in folder X"""
    print("🎬 Enhanced AI-Powered Vertical Video Generator")
    print("=" * 50)
    
    generator = EnhancedVideoGenerator()
    
    input_folder = "X"
    output_folder = "Y"
    
    # Ensure folders exist
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder '{input_folder}' not found!")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_folder, filename))
    
    if not image_files:
        print(f"❌ No image files found in '{input_folder}'!")
        print(f"   Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) to process")
    print(f"📤 Output folder: {output_folder}")
    print()
    
    # Process each image
    generated_videos = []
    for image_path in image_files:
        try:
            output_path = generator.process_image(image_path, output_folder)
            generated_videos.append(output_path)
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Video Generation Complete!")
    print(f"✅ Successfully generated {len(generated_videos)} video(s)")
    print(f"📁 All videos saved to folder: {output_folder}")
    print("\n📋 Generated videos:")
    for video in generated_videos:
        print(f"   • {os.path.basename(video)}")
    
    print("\n🎯 Features included:")
    print("   • Vertical format (1080x1920) optimized for social media")
    print("   • Animated person with subtle movements")
    print("   • Falling dollar bills with rotation effects")
    print("   • Dynamic sparkle and glow effects") 
    print("   • Motivational text overlays")
    print("   • Uplifting background music")
    print("   • High-quality MP4 output")

if __name__ == "__main__":
    main()