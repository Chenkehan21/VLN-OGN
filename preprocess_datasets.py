import gzip
import json


path = '/data/ckh/VLN-OGN/data/datasets/VLNCE/val_unseen/val_unseen.json.gz'
with gzip.open(path, 'r') as f:
    vln_val_data = json.loads(f.read().decode('utf-8'))['episodes']

path = '/data/ckh/VLN-OGN/data/datasets/VLNCE/val_unseen/scenes.txt'
with open(path, 'r') as f:
    scenes = f.readlines()
scenes = [item.strip() for item in scenes]
data = {key: [] for key in scenes}

for item in vln_val_data:
    scene = item['scene_id'][-15:-4]
    data[scene].append(item)

for key, value in data.items():
    info = {"episodes":value}
    json_str = json.dumps(info)
    with gzip.open('/data/ckh/VLN-OGN/data/datasets/VLNCE/val_unseen/content/%s_episodes.json.gz'%key, 'wt', encoding='utf-8') as gz_file:
        gz_file.write(json_str)

# import pdb;pdb.set_trace()