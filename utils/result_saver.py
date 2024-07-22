import os
import pickle
try:
    from .map import Map
except ImportError:
    from map import Map


def get_output_folder_name(map: Map) -> str:
    '''
    Get the output folder path for the simulation result

    Parameters:
    - map: The map object

    Returns:
    - str, Path to the folder to save the result
    '''
    t_lim           = map.t_lim
    spatial_density = map.spatial_density
    T0              = map.T0
    v0              = map.v0
    relative_mass   = map.ptcl_relative_mass

    output_folder_path = f'sim_{t_lim}_T0_{T0}_v0_{v0}_dsty_{spatial_density}_rltmass_{relative_mass:.2e}'
    return output_folder_path


def save_result(map: Map, output_folder: str = './saves') -> None:
    '''
    Save the result of the simulation

    Parameters:
    - map: The map object
    - output_folder: str, Path to the folder to save the result
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    keyframes = map.keyframes

    keyframes_dir = os.path.join(output_folder, get_output_folder_name(map))
    if not os.path.exists(keyframes_dir):
        os.makedirs(keyframes_dir)
    
    keyframes_file = os.path.join(keyframes_dir, 'keyframes.pkl')
    with open(keyframes_file, 'wb') as f:
        pickle.dump(keyframes, f)
    print(f'Saved keyframes to {keyframes_file}')

