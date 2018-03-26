# Database structure

MongoDB has databases, each of which can contain collections. 
Collections contain documents, which are just json objects with unique
"_id".

Songbird database: `metadata`
Collections: 
- manage: lists what playlist files from the Spotify dataset the database contains
- songInfo: contains all the metadata of songs that appear in the RecSys Spotify dataset
- myCollection: for prototyping and experimenting around with

### songInfo structure: 
Each document in songInfo has an "_id" corresponding to the song's uri. 
Abbreviated example (see Appendix section for full example):

````
{ 
        "_id" : "5IbCV9Icebx8rR6wAp5hhP",
        "tracks" : { [metadata from track category]
	},
        "audio-features" : { [metadata from audio-features category]
	}
}
````

# Accessing MongoDB using Python
[Reference](https://realpython.com/introduction-to-mongodb-and-python/)
Starter code
````
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017') # adjust for ip of database and port number
database = client['database_name'] # for us database_name is 'metadata'
collection = database['collection_name'] # for us 'collection_name' could be 'songInfo'
````

Update/create a document
````
collection.update_one({"_id": uri}, # query document with _id = uri
	{
            "$set": {category: data} # properties you want to change
        }, upsert=True) # upsert=True means create document if it doesn't exist
````

Get a document
````
cursor = collection.find_one({"_id": uri})
# cursor is a json dict representing the document (see "SongInfo structure" for example of a document)
````


# Appendix
SongInfo collection example document:
{ 
        "_id" : "5IbCV9Icebx8rR6wAp5hhP",
        "tracks" : {
                "album" : {
                        "album_type" : "album",
                        "artists" : [
                                {
                                        "external_urls" : {
                                                "spotify" : "https://open.spotify.com/artist/1SaAaqb
o2pHpm2ovncyGFU"
                                        },
                                        "href" : "https://api.spotify.com/v1/artists/1SaAaqbo2pHpm2o
vncyGFU",
                                        "id" : "1SaAaqbo2pHpm2ovncyGFU",
                                        "name" : "Original Cast",
                                        "type" : "artist",
                                        "uri" : "spotify:artist:1SaAaqbo2pHpm2ovncyGFU"
                                }
                        ],
                        "available_markets" : [
                                "CA",
                                "US"
                        ],
                        "external_urls" : {
                                "spotify" : "https://open.spotify.com/album/3ULJeOMgroG27dpn27MDfS"
                        },
                        "href" : "https://api.spotify.com/v1/albums/3ULJeOMgroG27dpn27MDfS",
                        "id" : "3ULJeOMgroG27dpn27MDfS",
                        "images" : [
                                {
                                        "height" : 640,
                                        "url" : "https://i.scdn.co/image/92c3def2007e4eddee1dc4cf8fc
6b7a1e4aa2d19",
                                        "width" : 629
                                },
                                {
                                        "height" : 300,
                                        "url" : "https://i.scdn.co/image/ad49d11685895fc5ecc9811ba42
25712fd68074d",
                                        "width" : 295
                                },
                                {
                                        "height" : 64,
                                        "url" : "https://i.scdn.co/image/5f12805346f5ca13fbc762ff012
8653579dd7ec9",
                                        "width" : 63
                                }
                        ],
                        "name" : "The Little Mermaid: Original Broadway Cast Recording",
                        "release_date" : "2008-01-01",
                        "release_date_precision" : "day",
                        "type" : "album",
                        "uri" : "spotify:album:3ULJeOMgroG27dpn27MDfS"
                },
                "artists" : [
                        {
                                "external_urls" : {
                                        "spotify" : "https://open.spotify.com/artist/3TymzPhJTMyupk7
P5xkahM"
                                },
                                "href" : "https://api.spotify.com/v1/artists/3TymzPhJTMyupk7P5xkahM",
                                "id" : "3TymzPhJTMyupk7P5xkahM",
                                "name" : "Original Broadway Cast - The Little Mermaid",
                                "type" : "artist",
                                "uri" : "spotify:artist:3TymzPhJTMyupk7P5xkahM"
                        }
                ],
                "available_markets" : [
                        "CA",
                        "US"
                ],
                "disc_number" : 1,
                "duration_ms" : 154506,
                "explicit" : false,
                "external_ids" : {
                        "isrc" : "USWD10733455"
                },
                "external_urls" : {
                        "spotify" : "https://open.spotify.com/track/5IbCV9Icebx8rR6wAp5hhP"
                },
                "href" : "https://api.spotify.com/v1/tracks/5IbCV9Icebx8rR6wAp5hhP",
                "id" : "5IbCV9Icebx8rR6wAp5hhP",
                "name" : "Fathoms Below - Broadway Cast Recording",
                "popularity" : 38,
                "preview_url" : null,
                "track_number" : 2,
                "type" : "track",
                "uri" : "spotify:track:5IbCV9Icebx8rR6wAp5hhP"
        },
        "audio-features" : {
                "danceability" : 0.623,
                "energy" : 0.263,
                "key" : 2,
                "loudness" : -15.612,
                "mode" : 1,
                "speechiness" : 0.0558,
                "acousticness" : 0.774,
                "instrumentalness" : 0,
                "liveness" : 0.147,
                "valence" : 0.645,
                "tempo" : 112.199,
                "type" : "audio_features",
                "id" : "5IbCV9Icebx8rR6wAp5hhP",
                "uri" : "spotify:track:5IbCV9Icebx8rR6wAp5hhP",
                "track_href" : "https://api.spotify.com/v1/tracks/5IbCV9Icebx8rR6wAp5hhP",
                "analysis_url" : "https://api.spotify.com/v1/audio-analysis/5IbCV9Icebx8rR6wAp5hhP",
                "duration_ms" : 154507,
                "time_signature" : 1
        }
}
