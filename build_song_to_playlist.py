import numpy as np
import scipy.sparse
import requests, time
import csv, json, sys, pickle
import base64, os.path
import pdb

try : 
    input_path = str(sys.argv[1])
    output_file = str(sys.argv[2])
    dimensions = str(sys.argv[3])
    rows, cols = dimensions.split(",")
    rows = int(rows)
    cols = int(cols)
except : 
    print("Usage:", sys.argv[0], "[directory that files are in] [output file] [number of playlists],[overestimate of number of songs]")
    print("Example: python3", sys.argv[0], "data matrix.npz 1000000,1000000000")
    quit()


# TODO: if the matrix is taking up too much space, change 'zeros' to a sparse matrix representation
theMatrix = np.zeros((rows, cols), dtype=bool) # rows are playlists, cols are songs
index_to_uri = {} # stores uri to index of each song
latest_index = -1

def getIndex(uri) : 
    global latest_index
    if uri in index_to_uri : 
        return index_to_uri[uri]
    else : 
        latest_index += 1
        index_to_uri[uri] = latest_index
        return latest_index

# removes any text before last occurence of ':' character
# such as spotify URI headers like "spotify:track:"
def clean_uri(dirty_uri, wantClean=True) : 
    if not wantClean : 
        return dirty_uri
    reversed_uri = dirty_uri[::-1]
    slice_index = reversed_uri.index(':')
    reversed_clean = reversed_uri[:slice_index]
    return reversed_clean[::-1]

# input: string of json slice as described in https://recsys-challenge.spotify.com/readme
# returns a list of URIs 
# removes the spotify:track: header if plz_clean_uri is true
def single_file(json_slice_file, plz_clean_uri=True) : 
    try : 
        the_json = json.load(open(input_path + '/' + json_slice_file))
    except : 
        print(json_slice_file)
        raise()
    playlists = the_json['playlists']
    track_uri_list = []
    for playlist in playlists : 
        playlist_id = playlist['pid']
        tracks = playlist['tracks']
        for track in tracks : 
            uri = clean_uri(track['track_uri'])
            index = getIndex(uri)
            if playlist_id > len(theMatrix) : 
                print("Not enough rows to store", playlist_id, "playlists")
            if index > theMatrix.shape[1] : 
                print("Not enough columns to store", index, "songs")
            theMatrix[playlist_id][index] = 1

theMatrix = theMatrix.T

file_list = os.listdir(input_path)
for f in file_list : 
    if f != ".DS_Store" : 
        single_file(f)

np.savez_compressed(output_file, arr_0=theMatrix, index_to_song=index_to_uri)
