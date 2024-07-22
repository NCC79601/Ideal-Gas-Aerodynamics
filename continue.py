import pickle
from utils.map import Map
from utils.visualize import make_video
from utils.result_saver import save_result

mp = Map('./config/config.json')

filename = './saves/sim_t_0.5_T0_298.15_v0_1000_dsty_0.03_rltmass_2.88e+01/keyframes_00.pkl'

# get the last keyframe of the previous simulation
try:
    with open(filename, 'rb') as f:
        pre_keyframes = pickle.load(f)
    last_keyframe = pre_keyframes[-1]
    mp.load_keyframe(last_keyframe)
    print('Loaded the last keyframe of the previous simulation.')
except FileNotFoundError:
    print('No previous simulation result found. Start a new simulation.')

# Resume simulation
mp.simulate()

print('Simulation complete.')

save_result(mp, output_folder='./saves')

print('Keyframes saved.')

print('Generating video...')

make_video(mp, output_folder='./saves', temp_folder='./temp_frames')