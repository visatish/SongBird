

import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import animation, rc
from IPython.display import HTML, Image

random.seed(101)#1 189 17 420 7 2 3 5 10 15 20 25 100 101 102
np.random.seed(101)
playlist_set_path= 'playlist_set.json'
playlist_data = json.load(open(playlist_set_path))
playlist_dir = "out/"

def parsePlaylists():
    playlists  = []
    all_fnames = os.        listdir(playlist_dir)
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


def train(feature_matrix, num_songs,num_iters,playlists,learning_rate,playlist_prob = 0.1):
    for i in range(num_iters):
        
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

def separateVectors(vec1,vec2,learning_rate,constant=1,dist =0.4):
    if np.linalg.norm(vec1-vec2)>dist:
        return vec1,vec2

    new_vec1 =(1+learning_rate/constant)*vec1-(learning_rate/constant)*vec2
    new_vec2 =(1+learning_rate/constant)*vec2-(learning_rate/constant)*vec1

    return normalize(new_vec1),normalize(new_vec2)

threshold = 150
total_iters = threshold*50+200*(570-threshold)
def update_lines(num,feature_matrix,lines):

    if num%10==0:
        print(num)
    if num<threshold:
        learning_rate=1-50*num/total_iters
        feature_matrix = train(feature_matrix,214,50,playlists,learning_rate)
        title.set_text('Playlist Clustering, iteration={}'.format(num*50))
    else:
        learning_rate=1-(50*threshold+(num-threshold)*200)/total_iters
        feature_matrix = train(feature_matrix,214,200,playlists,learning_rate)
        title.set_text('Playlist Clustering, iteration={}'.format(threshold*50+(num-threshold)*200))
    lines._offsets3d=(feature_matrix[:,0],feature_matrix[:,1],feature_matrix[:,2])
    #graph._offsets3d = (data.x, data.y, data.z)
    #lines.set_3d_properties(feature_matrix[:,2])
    #graph._offsets3d = (data.x, data.y, data.z)
    return lines

def determinePlaylist(feature_matrix):
    vec = np.zeros(feature_matrix.shape[0])
    for i in range(feature_matrix.shape[0]):
        vec[i] = (int)(np.random.choice(songToPlaylists(i),1))
    return vec

# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)

# # Fifty lines of random 3-D lines
# data = [Gen_RandLine(25, 3) for index in range(50)]

# # Creating fifty line objects.
# # NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# # Setting the axes properties
# ax.set_xlim3d([0.0, 1.0])
# ax.set_xlabel('X')

# ax.set_ylim3d([0. 1.0])
# ax.set_ylabel('Y')

# ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')

# ax.set_title('3D Test')

# # Creating the Animation object
# line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                    interval=50, blit=False)

# plt.show()
playlists = parsePlaylists()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
feature_matrix =randomFeatureMatrixInitialization(214,3)
playlistsPerSong = determinePlaylist(feature_matrix)
lines = ax.scatter(feature_matrix[:,0], feature_matrix[:,1], feature_matrix[:,2],c=playlistsPerSong,cmap="plasma")
# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')
title = ax.set_title('Playlist Clustering, iteration=0')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, frames=570, fargs=(feature_matrix,lines),
                                   interval=3, blit=False)
line_ani.save('animation3.gif', writer='imagemagick', fps=30)
plt.show()




