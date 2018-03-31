# reads in a file representing a dataset, ideally created by sampleDatabase.py
# file should be an npz file containing the keys 'uris' and 'features' that 
# correspond to each other
# 
# This program creates a SongDataset object that can be indexed into
# each index returns a Song object that contains uri and metadata
# 
# SongDataset can additionally be used to calculate pairwise IOUs
# between two sets of uris or Song objects.
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import numpy as np
import pdb

client = MongoClient('mongodb://localhost:27017')
database = client['metadata']
song_to_playlist_collection = database['song_to_playlist_copy']
FEATURE_COLLECTION = database['songInfo']


# example usage: sd = SongDataset("dataset.npz")
class SongDataset : 
    # input: string of file name containing dataset
    # dataset should have keys: 'uris' and 'features' containing corresponding song metadata
    def __init__(self, dataset_file_name) : 
        dataset = np.load(dataset_file_name)
        uris = dataset['uris']
        features = dataset['features']
        self.dataset = [Song(uri, features) for uri, features in zip(uris, features)]
    def __getitem__(self, indices) : 
        if isinstance(indices, list) : 
            return [self.dataset[i] for i in indices]
        return self.dataset[indices]
    def __len__(self) : 
        return len(self.dataset)
    def get_uri_list(self) : 
        return [song.uri for song in self.dataset]

    # takes in two lists of Songs or uris and returns their pairwise uris
    # example usage:
    # sd.get_IOU(sd[:10], sd[-10:])
    # [0.008272058823529412, 0.0014184397163120568, 0.007283264155376302, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    def get_IOU(self, lst1, lst2) : 
        if len(lst1) != len(lst2) : 
            raise Exception("lst1 and lst2 must be same length")
        if isinstance(lst1[0], Song) : 
            lst1 = [song.uri for song in lst1]
            lst2 = [song.uri for song in lst2]
        result = []
        for a, b in zip(lst1, lst2) : 
            result.append(getIOUPair(a, b))
        return result

class Song : 
    def __init__(self, uri, features) : 
        self.uri = uri
        self.features = np.array(features)

# a, b: track uris to get the IOU between
def getIOUPair(a, b) : 
    IOUPipeline = [
        {
            "$match": {
                "_id": { "$in": [a, b] }
            }
        },
        {
            "$group": {
                "_id": 0,
                "size1": { "$first": "$size" },
                "size2": { "$last": "$size" },
                "set1": { "$first": "$playlists" },
                "set2": { "$last": "$playlists" }
            }
        },
        {
            "$project": { 
                "intersection": { "$setIntersection": [ "$set1", "$set2" ] }, 
                "crude_union": { "$add": [ "$size1", "$size2" ] }, 
                "_id": 0 
            }
        },
        {
            "$project": {
                "IOU": {"$divide": [ 
                    {"$size": "$intersection"}, 
                        {"$subtract": [
                            "$crude_union",
                            {"$size": "$intersection"}
                        ] }
                    ]}
            }
        }
    ]
    result = song_to_playlist_collection.aggregate(IOUPipeline).next()
    return result['IOU']


