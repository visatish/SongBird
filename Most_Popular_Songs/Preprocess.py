import numpy as np
import json
import pickle 
import os
import operator

songdict = {}

file_index = 0
for filename in os.listdir("D:/data/"):
	the_json = json.load(open("D:/data/" + filename))
	print(filename)
	playlist_index = 0
	playlists = the_json['playlists']
	for playlist in playlists:
		track_index = 0
		for track in playlist["tracks"]:
			print("File #: {} out of {}, Filename: {}, Playlist: {}, Track: {}".format(file_index, 1000, filename, playlist_index, track_index))
			song_id = track["track_uri"][14:]
			if(song_id not in songdict):
				songdict[song_id] = 1
			else:
				songdict[song_id] += 1
			track_index += 1
		playlist_index += 1
	file_index += 1

print(len(songdict))

sorted_tuple = sorted(songdict.items(), key=operator.itemgetter(1), reverse = True)
sorted_tuple = sorted_tuple[0:20000]

with open('Data/Top20K.pickle', 'wb') as handle:
    pickle.dump(sorted_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)