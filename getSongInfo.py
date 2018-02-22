import requests, time
import csv, json, sys, pickle
import base64, os.path
import pdb

TIMEOUT = 60 # seconds
baseurl = "https://api.spotify.com/v1/"
client_id = "4790e38147b74e959fab0abe08e300ce"
client_secret = "761580a6eaf8451db3aabdc10ff46f07"

# get command line args
willPickle = True
try : 
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    categories = str(sys.argv[3])
    if len(sys.argv) > 4 : 
        if "json" in str(sys.argv[4]) : 
            willPickle = False
except IndexError : 
    print("Usage:", sys.argv[0], "[json input file] [output file] [comma separated list of categories NO SPACES AFTER COMMAS] [optional: json]")
    print("Example: python3", sys.argv[0], "mpd.slice.1000-1999.json slice.1000-1999.data.json tracks,audio-features,audio-analysis pickle")
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
            writeToFile()
            quit()
        else : 
            print(uri, r['error']['status'], r['error']['message'])
            return False
    return r

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
    
def writeToFile() : 
    print("Writing to file", output_file)
    if willPickle : 
        with open(output_file, 'wb') as outfile:
            pickle.dump(json_dict, outfile)
    else : 
        with open(output_file, 'w') as outfile:
            json.dump(json_dict, outfile)
    print("Successfully written")

# main 
uri_list = getURIList(input_file)
num_songs = len(uri_list)
print("Found", num_songs, "songs in file.")

print("Requesting access token for API calls")
success = getAuthorization()
if not success : 
    print("Spotify API sux")
    quit()
print("Access granted. Beginning download")

json_dict = {}
if os.path.isfile(output_file) : 
    if 'json' in output_file : 
        with open(output_file, 'r') as outfile : 
            json_dict = json.load(outfile)
    else : 
        with open(output_file, 'rb') as outfile : 
            json_dict = pickle.load(outfile)
num_downloaded = 0
changed = False
for i, uri in enumerate(uri_list) : 
    print("Downloading metadata for song", i+1, "out of", num_songs)
    if uri not in json_dict : 
        json_dict[uri] = {}
    for category in categoryList : 
        if category not in json_dict[uri] : 
            json_dict[uri][category] = getSongInfo(uri, category)
            num_downloaded += 1
            changed = True
        else : 
            print("song", uri, "category", category, "present. Skipping.")
    # write to file every 100 songs
    if num_downloaded % 100 == 0 and changed : 
        changed = False
        writeToFile()
writeToFile()
print("Downloaded information for", num_downloaded, "songs")
