import json
import os
data = json.load(open('song_set.json'))
out_dir = "out/"
DATA_DIR = 'Data' 
all_fnames = os.listdir(DATA_DIR)
for fname in all_fnames:
    curr_file = open(DATA_DIR+"/"+fname,"r") 
    new_file = open(out_dir+fname,"w") 
    ids = curr_file.read().splitlines()
    for curr_id in ids:

        index = data[curr_id]
        new_file.write(str(index)+"\n")
    curr_file.close() 
    new_file.close()