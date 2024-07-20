from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import os
import subprocess

def make_video(keyframes, width, height, walls, ptcl_radius, output_filename='output.mp4', temp_folder='temp_frames'):
    # 确保临时文件夹存在
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    # 获取所有时间点
    times = [kf['t'] for kf in keyframes]
    min_time, max_time = min(times), max(times)
    fps = 24  # 设置帧率
    duration = max_time - min_time
    total_frames = int(duration * fps) + 1

    print('Generating video frames...')

    for frame_idx in tqdm(range(total_frames)):
        t = min_time + frame_idx / fps
        closest_keyframe = min(keyframes, key=lambda kf: abs(kf['t'] - t))
        
        # 创建新图像
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 绘制粒子
        for pos in closest_keyframe['ptcl_pos']:
            left_up = (pos[0] - ptcl_radius, pos[1] - ptcl_radius)
            right_down = (pos[0] + ptcl_radius, pos[1] + ptcl_radius)
            draw.ellipse([left_up, right_down], fill='black')
        
        for wall in walls:
            draw.line([*wall[0], *wall[1]], fill='black', width=2)
        
        # 反转y轴
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 保存帧到临时文件夹
        frame_filename = os.path.join(temp_folder, f'frame_{frame_idx:04d}.png')
        img.save(frame_filename)
    
    # 使用ffmpeg合并图像为视频
    ffmpeg_cmd = f'ffmpeg -r {fps} -i {temp_folder}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_filename}'
    subprocess.run(ffmpeg_cmd, shell=True)
    
    # 清理临时文件
    for file_name in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file_name))
    os.rmdir(temp_folder)

    print(f'Video saved as {output_filename}')


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
