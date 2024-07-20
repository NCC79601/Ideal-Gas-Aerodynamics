from moviepy.editor import ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_video(keyframes, width, height, ptcl_radius, output_filename='output.mp4'):
    # Set the size of the canvas
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Get all the time points
    times = [kf['t'] for kf in keyframes]
    min_time, max_time = min(times), max(times)
    fps = 60  # Set the frame rate
    duration = max_time - min_time
    total_frames = int(duration * fps) + 1

    # Create a list of frames
    frames = []

    print('Generating video frames...')

    for frame_idx in tqdm(range(total_frames)):
        t = min_time + frame_idx / fps
        closest_keyframe = min(keyframes, key=lambda kf: abs(kf['t'] - t))
        
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        for pos in closest_keyframe['ptcl_pos']:
            circle = plt.Circle(pos, ptcl_radius, color='black')
            ax.add_patch(circle)
        
        fig.canvas.draw()
        
        # Add the current frame to the list of frames
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close(fig)
    
    # Use moviepy to combine the list of frames into a video
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_filename, codec='libx264')


if __name__ == '__main__':
    # example usage
    keyframes = [
        {"t": 0.0, "ptcl_pos": [(100, 100), (200, 200)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.1, "ptcl_pos": [(110, 110), (190, 190)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.2, "ptcl_pos": [(120, 120), (180, 180)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.3, "ptcl_pos": [(130, 130), (170, 170)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.4, "ptcl_pos": [(140, 140), (160, 160)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.5, "ptcl_pos": [(150, 150), (150, 150)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.6, "ptcl_pos": [(160, 160), (140, 140)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.7, "ptcl_pos": [(170, 170), (130, 130)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.8, "ptcl_pos": [(180, 180), (120, 120)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 0.9, "ptcl_pos": [(190, 190), (110, 110)], "ptcl_v": [(1, 1), (-1, -1)]},
        {"t": 1.0, "ptcl_pos": [(200, 200), (100, 100)], "ptcl_v": [(1, 1), (-1, -1)]},
    ]

    make_video(keyframes, 900, 600, 10)
