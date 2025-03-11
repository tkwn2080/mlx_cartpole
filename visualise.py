import cairo
import subprocess
import os
import numpy as np
from cartpole import DifficultyLevel  # Import DifficultyLevel for action mapping

def draw_frame(surface, state, action, step, difficulty):
    """
    Draw a single frame of the cartpole environment using PyCairo.
    
    Args:
        surface: Cairo ImageSurface to draw on
        state: List or array of [x, x_dot, theta, theta_dot]
        action: Integer representing the action taken
        step: Current step number
        difficulty: DifficultyLevel enum to determine action labels
    """
    # Extract state
    x, x_dot, theta, theta_dot = state
    
    # Map to pixel coordinates
    width, height = 800, 600
    track_y = 400
    cart_width, cart_height = 50, 20
    pole_length = 100  # Pole length in pixels for visualization
    cart_x = 400 + x * 125  # 125 pixels per meter, centered at x=400
    cart_y = track_y - cart_height / 2
    pole_end_x = cart_x + pole_length * np.sin(theta)
    pole_end_y = cart_y - pole_length * np.cos(theta)
    
    # Create context
    ctx = cairo.Context(surface)
    
    # Clear background to white
    ctx.set_source_rgb(1, 1, 1)  # White
    ctx.paint()
    
    # Draw track
    ctx.set_source_rgb(0, 0, 0)  # Black
    ctx.move_to(100, track_y)
    ctx.line_to(700, track_y)
    ctx.set_line_width(2)
    ctx.stroke()
    
    # Draw cart
    ctx.set_source_rgb(0, 0, 1)  # Blue
    ctx.rectangle(cart_x - cart_width / 2, cart_y - cart_height / 2, cart_width, cart_height)
    ctx.fill()
    
    # Draw pole
    ctx.set_source_rgb(1, 0, 0)  # Red
    ctx.move_to(cart_x, cart_y)
    ctx.line_to(pole_end_x, pole_end_y)
    ctx.set_line_width(5)
    ctx.stroke()
    
    # Draw text (step, action, state info)
    ctx.set_source_rgb(0, 0, 0)  # Black
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(20)
    
    # Map action to string based on difficulty
    if difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        action_str = ["Left", "Right", "Do Nothing"][action]
    else:
        action_str = ["Left", "Right"][action]
    
    ctx.move_to(10, 30)
    ctx.show_text(f"Step: {step}")
    ctx.move_to(10, 60)
    ctx.show_text(f"Action: {action_str}")
    ctx.move_to(10, 90)
    ctx.show_text(f"x: {x:.2f}, theta: {theta:.2f}")

def generate_video(states, actions, output_dir, trial_num, difficulty):
    """
    Generate an MP4 video from a sequence of states and actions.
    
    Args:
        states: List of state arrays [x, x_dot, theta, theta_dot]
        actions: List of action integers
        output_dir: Directory to save the video and temporary frames
        trial_num: Trial number for naming the video file
        difficulty: DifficultyLevel enum for action interpretation
    """
    # Create a directory for frames
    frame_dir = os.path.join(output_dir, f"frames_trial_{trial_num}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Generate each frame
    for step, (state, action) in enumerate(zip(states, actions)):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 800, 600)
        draw_frame(surface, state, action, step, difficulty)
        surface.write_to_png(os.path.join(frame_dir, f"frame_{step:04d}.png"))
    
    # Compile frames into MP4 using FFmpeg
    video_path = os.path.join(output_dir, f"trial_{trial_num}.mp4")
    cmd = [
        "ffmpeg",
        "-framerate", "25",  # 25 FPS for slower, observable playback
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up frame directory to save space
    import shutil
    shutil.rmtree(frame_dir)

if __name__ == "__main__":
    # Example usage for testing
    sample_state = [0.0, 0.0, 0.1, 0.0]  # [x, x_dot, theta, theta_dot]
    sample_action = 1
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 800, 600)
    draw_frame(surface, sample_state, sample_action, 0, DifficultyLevel.EASY)
    surface.write_to_png("sample_frame.png")
