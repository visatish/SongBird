import requests, time
import csv, json, sys, pickle
import base64
import pdb

baseurl = "https://api.spotify.com/v1/"
client_id = "4790e38147b74e959fab0abe08e300ce"
client_secret = "761580a6eaf8451db3aabdc10ff46f07"

# get command line args
willPickle = False
try : 
    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    categories = str(sys.argv[3])
    if len(sys.argv) > 4 : 
        if "pickle" in str(sys.argv[4]) : 
            willPickle = True
except IndexError : 
    print("Usage:", sys.argv[0], "[json input file] [output file] [comma separated list of categories NO SPACES AFTER COMMAS] [optional: pickle]")
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
    r = requests.post('https://accounts.spotify.com/api/token', data={"grant_type": "client_credentials"}, headers={'Authorization': 'Basic ' + encoded}).json()
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
    r = requests.get(baseurl + category + "/" + uri, headers=header).json()
    # handles errors
    while ('error' in r) : 
        # if api overloaded, waits appropriate time
        if (r['error']['status'] == 429) : # too many requests
            print(r)
            wait_time = r['error']['Retry-After']
            time.sleep(wait_time)
            r = requests.get(baseurl + category + "/" + uri, headers=header).json()
        elif (r['error']['status'] == 401) : # access token expired
            success = getAuthorization()
            if not success : 
                writeToFile()
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
getAuthorization()
print("Access granted. Beginning download")

json_dict = {}
for i, uri in enumerate(uri_list) : 
    print("Downloading metadata for song", i+1, "out of", num_songs)
    json_dict[uri] = {}
    for category in categoryList : 
        json_dict[uri][category] = getSongInfo(uri, category)
writeToFile()
