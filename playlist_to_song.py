import json
import sys
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017') # adjust for ip of database and port number
database = client['metadata'] # for us database_name is 'metadata'
collection = database['playlist_to_song'] # for us 'collection_name' could be 'songInfo'


data_json = json.load(open(sys.argv[0],'r')) # json dict
for playlist in data_json['playlists']: 
    
    collection.update_one({"_id": playlist['pid']}, # query document with _id = uri
     {
              "$set":  {'tracks': [track['track_uri'] for track in playlist['tracks']]}# properties you want to change
         }, upsert=True)