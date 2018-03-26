import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


playlist_set_path= 'playlist_set.json'
playlist_data = json.load(open(playlist_set_path))
playlist_dir = "out/"

def parsePlaylists():
    playlists = []
    all_fnames = os.listdir(playlist_dir)
    for fname in all_fnames:
        curr_file = open(playlist_dir+"/"+fname,"r") 
        ids = curr_file.read().splitlines()
        playlists.append(ids)
    return playlists

def randomVector(length):
    return np.random.rand(length)-0.5

def normalize(vec):
    return vec/np.linalg.norm(vec)

def randomFeatureMatrixInitialization(numSongs,num_dimensions):
    feature_matrix = np.zeros((numSongs,num_dimensions))
    for i in range(numSongs):
        feature_matrix[i,:] = normalize(randomVector(num_dimensions))
    return feature_matrix

def songToPlaylists(song):
    return playlist_data[str(song)]

def intermediateVectors(vec1,vec2,learning_rate,constant=1.0):
    new_vec1 =(1-learning_rate/constant)*vec1+(learning_rate/constant)*vec2
    new_vec2 =(1-learning_rate/constant)*vec2+(learning_rate/constant)*vec1

    return normalize(new_vec1),normalize(new_vec2)

def iterateViaPlaylist(feature_matrix, playlist,learning_rate,num_pairs):
    for _ in range(num_pairs):

        indices = random.sample(range(len(playlist)), 2)
        song1 = (int)(playlist[indices[0]])
        song2 = (int)(playlist[indices[1]])

        feat1 = feature_matrix[song1,:]
        feat2 = feature_matrix[song2,:]

        newfeat1,newfeat2 = intermediateVectors(feat1,feat2,learning_rate)

        feature_matrix[song1] = newfeat1
        feature_matrix[song2] = newfeat2

    return feature_matrix


def train(feature_matrix, num_songs,num_iters,playlists,playlist_prob = 0.1):
    for i in range(num_iters):
        learning_rate = 1.0-i /num_iters
        randVal = np.random.random()
        if randVal < playlist_prob:
            curr_playlist = playlists[random.randint(0,len(playlists)-1)]
            feature_matrix = iterateViaPlaylist(feature_matrix,curr_playlist,learning_rate,5)
        else:
            indices = random.sample(range(num_songs), 2)
            playlists1 = songToPlaylists(indices[0])
            playlists2 = songToPlaylists(indices[1])
            if len(list(set(playlists1) & set(playlists2)))==0:

                feat1 = feature_matrix[indices[0],:]
                feat2 = feature_matrix[indices[1],:]

                newfeat1,newfeat2 = separateVectors(feat1,feat2,learning_rate)

                feature_matrix[indices[0],:] = newfeat1
                feature_matrix[indices[1],:] = newfeat2

    return feature_matrix

def separateVectors(vec1,vec2,learning_rate,constant=1,dist =0.7):
    if np.linalg.norm(vec1-vec2)>dist:
        return vec1,vec2

    new_vec1 =(1+learning_rate/constant)*vec1-(learning_rate/constant)*vec2
    new_vec2 =(1+learning_rate/constant)*vec2-(learning_rate/constant)*vec1

    return normalize(new_vec1),normalize(new_vec2)

def determinePlaylist(feature_matrix):
    vec = np.zeros(feature_matrix.shape[0])
    for i in range(feature_matrix.shape[0]):
        vec[i] = (int)(np.random.choice(songToPlaylists(i),1))
    return vec
if __name__ == '__main__':
    vec1 = np.array([0,1,0])
    vec2 = np.array([1,0,0])
    #newVec1,newVec2=separateVectors(vec1,vec2,1)
    #print(np.linalg.norm(newVec1-newVec2))

    playlists = parsePlaylists()
    feature_matrix =randomFeatureMatrixInitialization(214,3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature_matrix[:,0],feature_matrix[:,1], feature_matrix[:,2], c='r', marker='o')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

    feature_matrix = train(feature_matrix,214,100000,playlists)
    playlistsPerSong = determinePlaylist(feature_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature_matrix[:,0],feature_matrix[:,1], feature_matrix[:,2], c=playlistsPerSong, marker='o')
    plt.show()


   

