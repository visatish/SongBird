import numpy as np
import json
import pickle 
import os
import operator

path = 'Data/Top20K.pickle'

def pickle_open(pathname):
	f =  open(pathname, 'rb')
	return pickle.load(f)

top_songs = pickle_open(path)

''' 
Loop through Million Playlists to find the Top 20k Songs in the Playlist

'''
song_dict = {}

for i in range(0, 20000):
	song_dict[top_songs[i][0]] = (i, top_songs[i][1])

file_index = 0
num_playlists = 0

playlist_data = []
for filename in os.listdir("D:/data/"):
	the_json = json.load(open("D:/data/" + filename))
	playlist_index = 0
	playlists = the_json['playlists']
	for playlist in playlists:
		track_index = 0
		track_list = []
		for track in playlist["tracks"]:
			#print("File #: {} out of {}, Filename: {}, Playlist: {}, Track: {}".format(file_index, 1000, filename, playlist_index, track_index))
			song_id = track["track_uri"][14:]
			if song_id in song_dict:
				track_list += [song_dict[song_id][0]]
			track_index += 1
		if len(track_list)>=8:
			print(len(playlist_data))
			playlist_data.append(track_list)
		playlist_index += 1
	file_index += 1

print(len(playlist_data))

with open('Data/PlaylistSequence.pickle', 'wb') as handle:
    pickle.dump(playlist_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
