from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import subprocess
import json
from utils.map import Map
from typing import Union
try:
    from .result_saver import get_output_folder_name
except ImportError:
    from result_saver import get_output_folder_name

def make_video(map_or_keyframes: Union[Map, list], output_folder='./saves', temp_folder='./temp_frames', config_file='./config/config.json') -> None:
    with open(config_file, 'r') as f:
        config = json.load(f)
    width       = config['map_size'][0]
    height      = config['map_size'][1]
    walls       = config['walls']
    ptcl_radius = config['ptcl_radius']
    video_fps   = config['video_fps']

    if isinstance(map_or_keyframes, Map):
        map = map_or_keyframes
        keyframes = map.keyframes
    else:
        keyframes = map_or_keyframes

    # check whether the temp folder exists
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    print('Generating video frames...')
    print(f'Total frames number: {len(keyframes)}')

    for frame_idx, keyframe in tqdm(enumerate(keyframes)):
        
        # create a new image
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # draw all particles
        for i, pos in enumerate(keyframe['ptcl_pos']):
            left_up = (pos[0] - ptcl_radius, pos[1] - ptcl_radius)
            right_down = (pos[0] + ptcl_radius, pos[1] + ptcl_radius)
            draw.ellipse([left_up, right_down], fill='black')

        # draw ignore area
        ignore_area_polygon = config['ignore_area_polygon']
        polygon_for_draw = [(x, y) for x,y in ignore_area_polygon]
        draw.polygon(polygon_for_draw, fill='white')
        
        # draw all walls
        for wall in walls:
            draw.line([*wall[0], *wall[1]], fill='black', width=wall[2])
        
        # invert y axis
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # save drawn keyframe images
        frame_filename = os.path.join(temp_folder, f'frame_{frame_idx:04d}.png')
        img.save(frame_filename)
    
    # 使用ffmpeg合并图像为视频
    if isinstance(map_or_keyframes, Map):
        output_file_dir = os.path.join(output_folder, get_output_folder_name(map))
        
    else:
        output_file_dir = os.path.join(output_folder, 'test')
    
    # check whether there exits any .mp4 file, if not, start numbering from 0
    file_index = 0
    while True:
        output_file_path = os.path.join(output_file_dir, f'output_{file_index:02d}.mp4')
        if not os.path.exists(output_file_path):
            break
        file_index += 1
    output_file_path = os.path.join(output_file_dir, f'output_{file_index:02d}.mp4')

    ffmpeg_cmd = f'ffmpeg -r {video_fps} -i {temp_folder}/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_file_path}'
    subprocess.run(ffmpeg_cmd, shell=True)
    
    # 清理临时文件
    for file_name in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file_name))
    os.rmdir(temp_folder)

    print(f'Video saved at {output_file_path}')


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
