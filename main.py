from utils.map import Map
from utils.visualize_efficient import make_video

mp = Map('./config/config.json')

mp.simulate()

print('Simulation complete.')

keyframes = mp.keyframes

make_video(keyframes, mp.width, mp.height, mp.walls, mp.ptcl_radius, output_filename='output.mp4')