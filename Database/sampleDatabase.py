# This file samples a prespecified number of samples from the database
# Gets the corresponding features and writes them to a prespecified file 
# (default: "dataset.npz")
import pymongo
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import time, datetime
import numpy as np
import pdb

TIMEOUT = 30 # in seconds
try : 
    client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=TIMEOUT)
except : 
    print("Cannot connect to database. Start it with command `sudo mongod`")
    # print(err)
    quit()
database = client['metadata']
collection = database['song_to_playlist_copy']
FEATURE_COLLECTION = database['songInfo']
TMP_COLLECTION_NAME = "tmp"

FEATURES = "danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,popularity,explicit,release_date"
features_list = FEATURES.split(',')

DEFAULT_VAL = 0

# num: number of uris to sample
# return a list of num uris representing tracks
def sampleTracks(num) :
    samplePipeline = [
            { "$sample": { "size": num} }
        ]
    try : 
        tracks = collection.aggregate(samplePipeline)
    except pymongo.errors.ServerSelectionTimeoutError : 
        print("Cannot connect to database. Start it with command `sudo mongod`")
        quit()
    uris = [t['_id'] for t in tracks]
    if len(uris) != num : 
        print("Warning: sampled", num, "tracks but database returned", len(uris), "tracks instead")
    return uris
    
# from https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-python-dictionaries-and-lists
# returns generator of all values matching specified key in dictionary
# key: wanted feature
# value: dict to search in
def find(key, value):
    if not isinstance(value, dict) :
        pdb.set_trace()
    if key in value.keys() :
        yield value[key]
    for k, v in value.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict) :
                    for result in find(key, d):
                        yield result
    return

# cleans the value outputted by the find method above
# returns a cleaned up numerical value, DEFAULT_VALUE otherwise
def cleanValue(raw_value_generator) : 
    try : 
        raw_value = next(raw_value_generator) 
    except StopIteration: 
        print("Warning: Feature does not exist")
        return DEFAULT_VAL
    if type(raw_value) == type(True) : # if value is boolean
        value = int(raw_value)
    elif type(raw_value) == type("") : # if value is string
        try : # converting to date
            value = time.mktime(datetime.datetime.strptime(raw_value, "%Y-%m-%d").timetuple())
        except ValueError :
            try :
                value = time.mktime(datetime.datetime.strptime(raw_value, "%Y-%m").timetuple())
            except ValueError :
                try :
                    value = time.mktime(datetime.datetime.strptime(raw_value, "%Y").timetuple())
                except ValueError :
                    print(raw_value, "Could not be processed")
                    value = DEFAULT_VAL
    else : 
        value = raw_value
    return value

# gets the features of a single song
# uri: a single uri
# output: nested dictionary/object/document as defined in the database
def getFeatures(uri) : 
    return FEATURE_COLLECTION.find({"_id": uri}).next()

def getFeatureMatrix(uris) : 
    rows = []
    for uri in uris : 
        featureDict = getFeatures(uri)
        features = [cleanValue(find(f, featureDict)) for f in features_list]
        rows.append(features)
    return rows

# num_training: number of desired training samples
# num_valid: number of desired validation samples
# output: 
#   uri list containing the order in which uris appear in the output
#   train_feature_matrix, train_feature_scaler: matrix of features and 
#        its corresponding sklearn.preprocessing.StandardScaler    
#   valid_feature_matrix, valid_feature_scaler: matrix of features and 
#        its corresponding sklearn.preprocessing.StandardScaler    
def buildDataset(num_samples, output_file="dataset.npz") : 
    # samples WITH replacement, but since dataset is so big the chance of 
    # duplicate entries is almost nil
    print("Sampling a total of", num_samples,  "tracks")
    uris = sampleTracks(num_samples)
    print("Done sampling")
    np.random.shuffle(uris)
    print("Constructing feature matrix")
    feature_matrix = getFeatureMatrix(uris)
    print("Done constructing")
    print("Writing to file", output_file)
    np.savez_compressed(output_file, uris=uris, features=feature_matrix)
    print("done")

# example usage
# buildDataset(100)

