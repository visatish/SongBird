import numpy as np
import scipy.sparse
import requests, time
import csv, json, sys, pickle
import base64, os.path
import pdb

try : 
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
except : 
    print("Usage:", sys.argv[0], "[input file] [output file]")
    print("Example: python3", sys.argv[0], "data matrix.npz 1000000,1000000000")
    quit()

# rows are songs, cols are playlists
print("loading", input_file) 
file_info = np.load(input_file)
print("loaded")

theMatrix = file_info['arr_0']
print("finished extracting data")

try : 
    songs_to_index = file_info['song_to_index']
except : 
    songs_to_index = {}

print("initializing iou matrix")
iou_matrix = np.eye(len(theMatrix))
print("done, beginning iou creation")
for song_i in range(len(theMatrix)) : 
    playlists_i = theMatrix[song_i]
    playlists_i = playlists_i.astype(int)
    print("Iteration", song_i, "of", len(theMatrix))
    for song_j in range(len(theMatrix)) : 
        if song_i != song_j : 
            playlists_j = theMatrix[song_j]
            playlists_j = playlists_j.astype(int)
            intersection = np.bitwise_and(playlists_i, playlists_j)
            union = np.bitwise_or(playlists_i, playlists_j)
            iou_matrix[song_i][song_j] = np.sum(intersection)/np.sum(union)

print("Done constructing matrix, writing to", output_file)
try : 
    slices = file_info['slices']
except : 
    slices = {}
np.savez_compressed(output_file, arr_0=iou_matrix, song_to_index=songs_to_index, slices=slices)
print("Done writing")
