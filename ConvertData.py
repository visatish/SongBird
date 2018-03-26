import pickle as pkl
import os
import json

DATA_DIR = 'Data' 
OUT_FILE_SSET_PKL = 'song_set.pkl'
OUT_FILE_SSET_JSON = 'song_set.json'
OUT_FILE_PSET_PKL = 'playlist_set.pkl'
OUT_FILE_PSET_JSON = 'playlist_set.json'

song_set = {}
playlist_set = {}
curr_id = 0

all_fnames = os.listdir(DATA_DIR)

playlists = []
for fname in all_fnames:
    with open(os.path.join(DATA_DIR, fname), 'r') as fhandle:
        playlists.append(fhandle.read().split('\n')) 

for p in range(len(playlists)):
    for s in playlists[p]:
        if s not in song_set.keys():
            song_set[s] = curr_id
            curr_id += 1
        if song_set[s] not in playlist_set.keys():
            playlist_set[song_set[s]] = [p]
        else:
            playlist_set[song_set[s]].append(p)

with open(OUT_FILE_SSET_PKL, 'wb') as fhandle:
    pkl.dump(song_set, fhandle)

with open(OUT_FILE_SSET_JSON, 'w') as fhandle:
    json.dump(song_set, fhandle)

with open(OUT_FILE_PSET_PKL, 'wb') as fhandle:
    pkl.dump(playlist_set, fhandle)

with open(OUT_FILE_PSET_JSON, 'w') as fhandle:
    json.dump(playlist_set, fhandle)