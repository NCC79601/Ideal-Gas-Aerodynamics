import json
from utils.map import Map
from utils.visualize import make_video
from utils.result_saver import save_result

mp = Map('./config/config.json')

mp.simulate()

print('Simulation complete.')

save_result(mp, output_folder='./saves')

print('Keyframes saved.')

print('Generating video...')

make_video(mp, output_folder='./saves', temp_folder='./temp_frames')