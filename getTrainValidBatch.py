from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import time, datetime
import numpy as np
import pdb

client = MongoClient('mongodb://localhost:27017')
database = client['metadata']
collection = database['song_to_playlist_copy']
FEATURE_COLLECTION = database['songInfo']
TMP_COLLECTION_NAME = "tmp"

FEATURES = "danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,popularity,explicit,release_date"
features_list = FEATURES.split(',')

DEFAULT_VAL = np.inf
# num: number of uris to sample
# return a list of num uris representing tracks
def sampleTracks(num) :
    samplePipeline = [
            { "$sample": { "size": num} }
        ]
    tracks = collection.aggregate(samplePipeline)
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
        return DEFAULT_VALUE
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
    return np.matrix(rows)

# unused atm, invoke whenever
# a, b: track uris to get the IOU between
def getIOU(a, b) : 
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
    result = collection.aggregate(IOUPipeline).next()
    return result['IOU']

# num_training: number of desired training samples
# num_valid: number of desired validation samples
# output: 
#   uri list containing the order in which uris appear in the output
#   train_feature_matrix, train_feature_scaler: matrix of features and 
#        its corresponding sklearn.preprocessing.StandardScaler    
#   valid_feature_matrix, valid_feature_scaler: matrix of features and 
#        its corresponding sklearn.preprocessing.StandardScaler    
def getTrainValidData(num_training, num_valid) : 
    # samples WITH replacement, but since dataset is so big the chance of 
    # duplicate entries is almost nil
    print("Sampling a total of", num_training + num_valid, "tracks")
    uris = sampleTracks(num_training + num_valid)
    print("Done sampling")
    # uris = ["7KXjTSCq5nL1LoYtL7XAwS", "1xznGGDReH1oQq0xzbwXa3", "7yyRTcZmCiyzzJlNzGC9Ol", "7BKLCZ1jbUBVqRi2FVlTVw"] # most popular tracks
    np.random.shuffle(uris)
    print("Constructing feature matrix")
    feature_matrix = getFeatureMatrix(uris)
    print("Done constructing")
    # iou_matrix = getIOUMatrix(uris)
    train_feature_matrix = feature_matrix[:num_training]
    train_feature_scaler = StandardScaler()
    train_feature_scaler.fit(train_feature_matrix)
    train_feature_scaler.mean = train_feature_scaler.mean_
    train_feature_scaler.var = train_feature_scaler.var_
    # train_iou_matrix = iou_matrix[:num_training]
    pdb.set_trace()

    valid_feature_matrix = feature_matrix[num_training:]
    valid_feature_scaler = StandardScaler()
    valid_feature_scaler.fit(valid_feature_matrix)
    valid_feature_scaler.mean = valid_feature_scaler.mean_
    valid_feature_scaler.var = valid_feature_scaler.var_
    # valid_iou_matrix = iou_matrix[:num_training]
    return uris, (train_feature_matrix, train_feature_scaler), (valid_feature_matrix, valid_feature_scaler)
    # return (train_feature_matrix, train_iou_matrix), (valid_feature_matrix, valid_iou_matrix)

# example usage
# uris, train_tuple, valid_tuple = getTrainValidData(100, 10)


######## UNUSED CODE FEEL FREE TO IGNORE #######
# # queries for iou for every pair
# def getIOUMatrix(uris) : 
#     mat = np.eye(len(uris))
#     k = 0
#     for i, a in enumerate(uris) : 
#         for j, b in enumerate(uris) : 
#             if i != j and mat[i][j] == 0 : 
#                 iou = getIOU(a, b)
#                 mat[i][j] = iou
#                 mat[j][i] = iou
#                 k += 1
#     print(k)
#     return mat

# # builds collection of pairs, creates IOU in database, then retrieves results
# def getIOUMatrix(uris) : 
#     mat = np.eye(len(uris))
#     k = 0
#     createURIPairs(uris)
#     cursor = generateIOUFromPairs()
#     for i, a in enumerate(uris) : 
#         for j, b in enumerate(uris) : 
#             if i != j and mat[i][j] == 0 : 
#                 mat[i][j] = iou
#                 mat[j][i] = iou
#                 k += 1
#     print(k)
#     return mat

# def createURIPairs(uris) : 
#     documents = []
#     for a in uris : 
#         for b in uris : 
#             doc = {"tracks": [a, b]}
#             documents.append(doc)
#     print("Created document pairs. Inserting to database")
#     database[TMP_COLLECTION_NAME].insert_many(documents)
#     print("Done inserting")
# 
# def generateIOUFromPairs() : 
#     pipeline = [
#         {"$lookup": {
#             "from": "song_to_playlist_copy",
#             "localField": "tracks",
#             "foreignField": "_id",
#             "as": "other"
#         }},
#         {
#             "$project": { 
#                 "obj1": {"$arrayElemAt": ["$other", 0]},
#                 "obj2": {"$arrayElemAt": ["$other", 1]},
#                 "_id": 0 
#             }
#         },
#         {
#             "$project": {
#                 "track1": "$obj1._id",
#                 "track2": "$obj2._id",
#                 "intersection": { "$setIntersection": [ "$obj1.playlists", "$obj2.playlists" ] }, 
#                 "crude_union": { "$add": [ "$obj1.size", "$obj2.size" ] }, 
#             }
#         },
#         {
#             "$match": {"intersection": {"$ne": null}}
#         },
#         {
#             "$project": {
#                 "track1": 1,
#                 "track2": 1,
#                 "IOU": {"$divide": [ 
#                     {"$size": "$intersection"}, 
#                         {"$subtract": [
#                             "$crude_union",
#                             {"$size": "$intersection"}
#                         ] }
#                     ]}
#             }
#         }
#     ]
#     return cursor = database[TMP_COLLECTION_NAME].aggregate(pipeline)
