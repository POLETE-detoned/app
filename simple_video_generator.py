#!/usr/bin/env python3
"""
Simplified AI-Powered Vertical Video Generator
Creates viral-style vertical videos with animated effects
"""

import os
import cv2
import numpy as np
import random
import math
from PIL import Image, ImageDraw
import time

def create_sample_frames():
    """Create sample animated frames for testing"""
    width, height = 1080, 1920
    fps = 30
    duration = 20  # seconds
    total_frames = int(duration * fps)
    
    frames = []
    
    print(f"Generating {total_frames} frames for {duration}s video...")
    
    for frame_num in range(total_frames):
        if frame_num % 60 == 0:
            print(f"Processing frame {frame_num}/{total_frames}")
        
        # Create base frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background
        
        # Add gradient background
        for i in range(height):
            intensity = int(30 + 50 * (i / height))
            frame[i, :] = [intensity - 10, intensity, intensity + 20]
        
        # Animate person figure
        person_x = width // 2
        person_y = height // 2
        
        # Walking animation
        walk_offset = math.sin(frame_num * 0.3) * 10
        bounce = abs(math.sin(frame_num * 0.2)) * 5
        
        # Head
        head_x = int(person_x + walk_offset)
        head_y = int(person_y - 100 - bounce)
        cv2.circle(frame, (head_x, head_y), 40, (220, 180, 140), -1)
        cv2.circle(frame, (head_x, head_y), 40, (255, 255, 255), 2)
        
        # Eyes
        cv2.circle(frame, (head_x - 15, head_y - 10), 5, (0, 0, 0), -1)
        cv2.circle(frame, (head_x + 15, head_y - 10), 5, (0, 0, 0), -1)
        
        # Smile
        cv2.ellipse(frame, (head_x, head_y + 10), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        
        # Body
        body_x = int(person_x + walk_offset)
        body_y = int(person_y + 20 - bounce)
        cv2.rectangle(frame, (body_x - 30, body_y - 60), (body_x + 30, body_y + 80), (100, 150, 200), -1)
        cv2.rectangle(frame, (body_x - 30, body_y - 60), (body_x + 30, body_y + 80), (255, 255, 255), 2)
        
        # Animated arms (raising for happiness)
        arm_raise = math.sin(frame_num * 0.1) * 30
        left_arm_y = int(body_y - 20 - arm_raise)
        right_arm_y = int(body_y - 20 - arm_raise)
        
        cv2.line(frame, (body_x - 30, body_y - 40), (body_x - 70, left_arm_y), (220, 180, 140), 15)
        cv2.line(frame, (body_x + 30, body_y - 40), (body_x + 70, right_arm_y), (220, 180, 140), 15)
        
        # Hands
        cv2.circle(frame, (body_x - 70, left_arm_y), 12, (220, 180, 140), -1)
        cv2.circle(frame, (body_x + 70, right_arm_y), 12, (220, 180, 140), -1)
        
        # Legs
        cv2.line(frame, (body_x - 15, body_y + 80), (body_x - 15, body_y + 150), (100, 150, 200), 20)
        cv2.line(frame, (body_x + 15, body_y + 80), (body_x + 15, body_y + 150), (100, 150, 200), 20)
        
        # Add falling dollar bills
        num_bills = 8
        for i in range(num_bills):
            bill_x = (i * 150 + frame_num * 3) % (width + 100) - 50
            bill_y = (frame_num * 6 + i * 200) % (height + 200) - 100
            
            if 0 <= bill_x < width - 60 and 0 <= bill_y < height - 30:
                # Create dollar bill
                cv2.rectangle(frame, (bill_x, bill_y), (bill_x + 60, bill_y + 30), (34, 139, 34), -1)
                cv2.rectangle(frame, (bill_x, bill_y), (bill_x + 60, bill_y + 30), (255, 255, 255), 2)
                cv2.putText(frame, '$', (bill_x + 20, bill_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add sparkles
        num_sparkles = 15
        for i in range(num_sparkles):
            spark_x = (frame_num * 2 + i * 123) % width
            spark_y = (frame_num + i * 456) % height
            intensity = int(128 + 127 * math.sin(frame_num * 0.1 + i))
            
            cv2.circle(frame, (spark_x, spark_y), 3, (255, 255, intensity), -1)
            cv2.circle(frame, (spark_x, spark_y), 6, (255, 255, intensity//2), 1)
        
        # Add motivational text
        progress = frame_num / total_frames
        if progress < 0.3:
            text = "MONEY FLOWS TO ME"
        elif progress < 0.7:
            text = "ABUNDANCE MINDSET"
        else:
            text = "SUCCESS IS COMING"
        
        # Text animation
        text_scale = 1.5 + math.sin(frame_num * 0.1) * 0.3
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_scale, 3)
        text_x = (width - text_width) // 2
        text_y = 150
        
        # Add text with outline
        cv2.putText(frame, text, (text_x-2, text_y-2), font, text_scale, (0, 0, 0), 5)
        cv2.putText(frame, text, (text_x, text_y), font, text_scale, (255, 255, 0), 3)
        
        frames.append(frame)
    
    return frames, fps

def create_video_with_ffmpeg(frames, fps, output_path):
    """Create video using OpenCV VideoWriter"""
    print(f"Creating video with {len(frames)} frames at {fps} fps...")
    
    height, width = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        if i % 60 == 0:
            print(f"Writing frame {i}/{len(frames)}")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")

def generate_background_audio(duration, output_path):
    """Generate simple background audio"""
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create uplifting chord progression
        frequencies = [261.63, 329.63, 392.00, 523.25]  # C, E, G, C
        music = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            phase_offset = i * np.pi / 4
            wave = 0.2 * np.sin(2 * np.pi * freq * t + phase_offset)
            envelope = np.abs(np.sin(2 * np.pi * t * 2))
            music += wave * envelope
        
        # Add bass
        bass_freq = 130.81
        bass = 0.3 * np.sin(2 * np.pi * bass_freq * t) * np.abs(np.sin(2 * np.pi * t))
        music += bass
        
        # Normalize
        music = music / np.max(np.abs(music)) * 0.7
        music_16bit = (music * 32767).astype(np.int16)
        
        # Save as WAV
        import wave
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(music_16bit.tobytes())
        
        print(f"Audio saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Could not generate audio: {e}")
        return False

def combine_video_audio(video_path, audio_path, output_path):
    """Combine video and audio using ffmpeg"""
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Final video with audio saved to: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Could not combine video and audio: {e}")
        return False

def main():
    """Main function"""
    print("🎬 Starting Vertical Video Generation...")
    
    # Create output directory
    os.makedirs("Y", exist_ok=True)
    
    # Generate frames
    frames, fps = create_sample_frames()
    
    # Create video
    temp_video = "temp_video.mp4"
    create_video_with_ffmpeg(frames, fps, temp_video)
    
    # Generate audio
    duration = len(frames) / fps
    temp_audio = "temp_audio.wav"
    audio_created = generate_background_audio(duration, temp_audio)
    
    # Combine video and audio
    final_output = "Y/animated_video.mp4"
    if audio_created and os.path.exists(temp_audio):
        if combine_video_audio(temp_video, temp_audio, final_output):
            # Clean up temp files
            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        else:
            # If combining failed, just use video without audio
            import shutil
            shutil.move(temp_video, final_output)
            print("Video saved without audio")
    else:
        # If audio generation failed, just use video
        import shutil
        shutil.move(temp_video, final_output)
        print("Video saved without audio")
    
    print(f"🎉 Video generation complete!")
    print(f"📁 Output: {final_output}")
    print(f"📏 Resolution: 1080x1920 (vertical)")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🎯 Features: Animated person, falling dollars, sparkles, motivational text")

if __name__ == "__main__":
    main()