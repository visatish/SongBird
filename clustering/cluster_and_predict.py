
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K_MEANS_K = 14

data = np.load('2162686_songs_04_11_18.npz')
X = data['features']
uris = data['uris']

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Finished scaling")

# TRAINING
kmeans = KMeans(n_clusters=K_MEANS_K) 
kmeans.fit(X_scaled)
neighbors = NearestNeighbors(n_neighbors=2)
neighbors.fit(X_scaled)

# TESTING
# a list of matrices of song vectors corresponding to a playlist
X_test = buildTestFromPlaylists(num_playlists=1)

playlist_predictions = []
for playlist in X_test
    num_songs = len(playlist)
    global_clusters = kmeans.predict(playlist)
    groupedSongs = groupSongsByCluster(global_clusters, playlist)

    playlist_centroids = {cluster_idx: np.average(songs, axis=0) for (cluster_idx, songs) in groupedSongs.items()}

    predicted_songs = []
    for centroid in playlist_centroids : 
        cluster_size = len(groupedSongs[centroid])
        indices = neighbors.kneighbors(centroid, n_neighbors=cluster_size, return_distance=False)
        indices = sum(indices, []) # some cleaning goin on
        predicted_songs.extend(indices)
    # convert indices to uris
    predicted_songs = [uris[i] for i in predicted_songs]
    playlist_predictions.append(predicted_songs)




###### methods for the algo #######
# clusters indices and songs lists of same length
# cluster[i] contains the cluster that songs[i] belongs to
# returns a dict with keys = cluster index, values = list of songs
# that belong to that cluster
# Yes I know it's basically a list. If you have problems take it up
# with my lawyer
def groupSongsByCluster(cluster_idxs, songs) : 
    if len(cluster_idxs) != len(songs) : 
        raise Exception("Length of cluster_idxs and songs must be equal")
    groupedSongs = {}
    # initialize dict
    for c in set(cluster_idxs) : 
        groupedSongs[c] = []

    for i in range(len(cluster_idxs)) : 
        groupedSongs[cluster_idxs[i]] = songs[i]
    return groupedSongs

# playlist_list is a list of playlist ids
# if left empty, will sample num_playlists random playlists
def buildTestFromPlaylists(playlist_list=[], num_playlists=1) : 
    global X_scaled, uris
    if len(playlist_list) == 0 : 
        random_playlists = db.getCollection('playlist_to_song').aggregate(
               [ { $sample: { size: num_playlists } } ]
        )
        playlists = list(random_playlists)
    else : 
        # get playlists by id

    all_playlists = []
    for playlist in playlists : 
        playlist_matrix = []
        for uri in playlist['tracks'] : 
            song_vec = X_scaled[uris.indexOf(uri)]
            playlist_matrix.append(song_vec)
        all_playlists.append(np.matrix(playlist_matrix))
    return all_playlists
