import gzip
import json
import bz2
import _pickle as cPickle


path = '/data/ckh/Object-goal-Navigation/data/datasets/objectnav/gibson/v1.1/val/content/Collierville_episodes.json.gz'
with gzip.open(path, 'r') as f:
    eps_data = json.loads(f.read().decode('utf-8'))

'''
type(eps_data): dict
len(eps_data)=1
eps_data.keys(): dict_keys(['episodes'])
type(eps_data['episodes']): list
len(eps_data['episodes'])=200

eps_data['episodes'][0]={
                            'episode_id': 0, 
                            'scene_id': 'gibson_semantic/Collierville.glb', 
                            'start_position': [-0.3094837963581085, 0.026745080947875977, -3.483508586883545], 
                            'start_rotation': [-0.22018953669540486, 0.0, 0.9754571071707168, 0.0], [x,y,z,w]
                            'object_category': 'toilet', 
                            'object_id': 4, 
                            'floor_id': 0
                        }
'''


path = '/data/ckh/VLN-OGN/data/datasets/VLNCE/val_unseen/content/2azQ1b91cZZ_episodes.json.gz'
with gzip.open(path, 'r') as f:
    vln_eps_data = json.loads(f.read().decode('utf-8'))


dataset_info_file = '/data/ckh/Object-goal-Navigation/data/datasets/objectnav/gibson/v1.1/val/val_info.pbz2'
with bz2.BZ2File(dataset_info_file, 'rb') as f:
    dataset_info = cPickle.load(f)

'''
type(dataset_info): dict
dataset_info.keys(): dict_keys(['Collierville', 'Corozal', 'Darden', 'Markleeville', 'Wiconisco'])
type(dataset_info['Collierville']): dict
dataset_info['Collierville'].keys(): dict_keys([0])
type(dataset_info['Collierville'][0]): dict
dataset_info['Collierville'][0].keys(): dict_keys(['floor_height', 'sem_map', 'origin'])

type(dataset_info['Collierville'][0]['floor_height']): <class 'float'>
type(dataset_info['Collierville'][0]['sem_map']): <class 'numpy.ndarray'>; shape: (16, 149, 239)
type(dataset_info['Collierville'][0]['origin']): <class 'numpy.ndarray'>; shape: (2,) e.g. array([-911, -431])
'''


path = '/data/ckh/Object-goal-Navigation/data/datasets/objectnav/gibson/v1.1/val/val.json.gz'
with gzip.open(path, 'r') as f:
    val_data = json.loads(f.read().decode('utf-8'))
'''
val_data is empty
'''

path = '/data/ckh/VLN-OGN/data/datasets/VLNCE/val_unseen/val_unseen.json.gz'
with gzip.open(path, 'r') as f:
    vln_val_data = json.loads(f.read().decode('utf-8'))
    
import pdb;pdb.set_trace()