import json
from utils.map import Map
from utils.visualize_efficient import make_video
import pickle

mp = Map('./config/config.json')

mp.simulate()

print('Simulation complete.')

keyframes = mp.keyframes

keyframes_file = './keyframes.pkl'
with open(keyframes_file, 'wb') as f:
    pickle.dump(keyframes, f)

print('Keyframes saved.')

print('Generating video...')

make_video(keyframes, mp.width, mp.height, mp.walls, mp.ptcl_radius, output_filename='output.mp4')