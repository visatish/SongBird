# This program splits an existing pickled or json file containing
# song into jsonArrays to be imported to MongoDB
import pickle, json

theDict = pickle.load(open('800mb_track.pickle', 'rb'))
num_songs = len(theDict)
print("file loaded, writing", num_songs, "songs")

basename = "output_"

max_buffer = 15000
song_buffer = [] # hold max of 15000 songs
num_buffers = 0
def writeToFile() : 
    global song_buffer, num_buffers
    with open(basename + str(num_buffers) + '.array', 'w') as thefile : 
        json.dump(song_buffer, thefile) 
    # thefile.write(pprint.pformat(song_buffer))
    song_buffer = []
    num_buffers += 1
    print("Wrote up to song", num_buffers * max_buffer, "of", num_songs)

for song in theDict : 
    if len(song_buffer) == max_buffer : 
        writeToFile()
    theDict[song]["_id"] = song
    song_buffer.append(theDict[song])
writeToFile()
