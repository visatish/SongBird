import requests, time
from pymongo import MongoClient
import csv, json, sys, pickle
import base64, os.path
import pdb

TIMEOUT = 60 # seconds
baseurl = "https://api.spotify.com/v1/"
client_id = "4790e38147b74e959fab0abe08e300ce"
client_secret = "761580a6eaf8451db3aabdc10ff46f07"

DATABASE = "metadata"
COLLECTION = "songInfo"
CHUNK_SIZE = 50
# get command line args
willPickle = True
try : 
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    categories = str(sys.argv[3])
    if len(sys.argv) > 4 : 
        if "pickle" in str(sys.argv[4]) : 
            willPickle = True
except IndexError : 
    print("Usage:", sys.argv[0], "[json input file] [mongourl] [comma separated list of categories NO SPACES AFTER COMMAS]")
    print("Example: python3", sys.argv[0], "mpd.slice.1000-1999.json mongodb://localhost:27017 tracks,audio-features,audio-analysis")
    quit()

categoryList = []
if "," in categories : 
    categoryList = categories.split(",")
else : 
    categoryList.append(categories)

def stringToBase64(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')

def getAuthorization() : 
    global TOKEN
    decoded = client_id + ":" + client_secret
    encoded = stringToBase64(decoded)
    try : 
        r = requests.post('https://accounts.spotify.com/api/token', data={"grant_type": "client_credentials"}, headers={'Authorization': 'Basic ' + encoded}, timeout=TIMEOUT).json()
    except requests.exceptions.Timeout: 
        print("request for token timed out")
        return False
    if 'error' in r : 
        print(r)
        return False
    if 'access_token' in r : 
        TOKEN = r['access_token']
        return True
    else : 
        print(r)
        return False


# returns json with desired category info for a SINGLE song
def getSongInfo(uri, category, several=False) : 
    header = {"'Accept'": "application/json", 'Authorization': 'Bearer ' + TOKEN}
    try : 
        r = requests.get(baseurl + category + "/" + uri, headers=header, timeout=TIMEOUT).json()
    except requests.exceptions.Timeout: 
        print("Request for uri", uri, "category", category, "timed out. Skipping.")
        return 'timed out'
    # handles errors
    while ('error' in r) : 
        # if api overloaded, waits appropriate time
        if (r['error']['status'] == 429) : # too many requests
            print(r)
            wait_time = r['error']['Retry-After']
            time.sleep(wait_time)
            try : 
                r = requests.get(baseurl + category + "/" + uri, headers=header, timeout=TIMEOUT).json()
            except requests.exceptions.Timeout:
                print("Request for uri", uri, "category", category, "timed out. Skipping.")
                return 'timed out'
        elif (r['error']['status'] == 401) : # access token expired
            print("Authorization token expired, terminating")
            quit()
        else : 
            print(uri, r['error']['status'], r['error']['message'])
            return False
    return r

# returns json with desired category info for a HUNDRED song
def getHundredSongInfo(uri, category, several=False) : 
    header = {"'Accept'": "application/json", 'Authorization': 'Bearer ' + TOKEN}
    try : 
        uris_string = ','.join(uri)
        r = requests.get(baseurl + category + "?ids=" + uris_string, headers=header, timeout=TIMEOUT).json()
    except requests.exceptions.Timeout: 
        print("Request for uri", uri, "category", category, "timed out. Skipping.")
        return 'timed out'
    # handles errors
    while ('error' in r) : 
        # if api overloaded, waits appropriate time
        if (r['error']['status'] == 429) : # too many requests
            print(r)
            wait_time = r['error']['Retry-After']
            time.sleep(wait_time)
            try : 
                r = requests.get(baseurl + category + "/" + uri, headers=header, timeout=TIMEOUT).json()
            except requests.exceptions.Timeout:
                print("Request for uri", uri, "category", category, "timed out. Skipping.")
                return 'timed out'
        elif (r['error']['status'] == 401) : # access token expired
            print("Authorization token expired, terminating")
            quit()
        else : 
            print(uri, r['error']['status'], r['error']['message'])
            return False
    category = category.replace("-", "_")
    return r[category]
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
def getURIList(json_slice_file, plz_clean_uri=True) : 
    the_json = json.load(open(json_slice_file))
    playlists = the_json['playlists']
    track_uri_list = []
    for playlist in playlists : 
        tracks = playlist['tracks']
        track_uri_list += [clean_uri(t['track_uri'], plz_clean_uri) for t in tracks]
    return track_uri_list
    
first = True
def writeToDB(uri_list, data_list, category) : 
    global collection, first
    for uri, data in zip(uri_list, data_list) : 
        if first : 
            print(uri)
            first = False
        collection.update_one({"_id": uri}, {
            "$set": {category: data}
        }, upsert=True)
# main 
uri_list = getURIList(input_file)
num_songs = len(uri_list)
print("Found", num_songs, "songs in file.")

print("Connecting to database at", output_file)
client = MongoClient(output_file)
db = client[DATABASE]
collection = db[COLLECTION]
print("Connected")

def contains(uri) : 
    cursor = collection.find_one({"_id": uri})
    return bool(cursor)
num_downloaded = 0
changed = False
print("Removing already downloaded songs")
# remove already downloaded songs
uri_list = [uri for uri in uri_list if not contains(uri)]
print(len(uri_list), "songs to be downloaded")
if len(uri_list) == 0 : 
    quit()

print("Requesting access token for API calls")
success = getAuthorization()
if not success : 
    print("Spotify API sux")
    quit()
print("Access granted. Beginning download")
print("token is", TOKEN)

# split into chunks of 100 songs
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
gen = chunks(uri_list, CHUNK_SIZE)
uri_chunks = list(gen)
num_chunks = len(uri_chunks)
print("Split into", num_chunks, "chunks")

for i in range(len(uri_chunks)) : 
    print("Downloading metadata for chunk", i+1, "out of", num_chunks, "in file", input_file)
    for category in categoryList : 
        hundred_metadata = getHundredSongInfo(uri_chunks[i], category)
        writeToDB(uri_chunks[i], hundred_metadata, category)
# record that the input file has been processed
db['manage'].insert({"filename": input_file})
with open('processed_files.sh', 'a') as outfile : 
    outfile.write(input_file + '\n')
print("DONE")
