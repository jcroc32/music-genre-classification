import numpy as np

# split song into sections of time length given
def splitsong(song, sr, time_length):
	x = []
	step = sr*time_length
	for s in [song[i:i+step] for i in range(0, song.shape[0], step)]:
		x.append(s)
	return np.asarray(x)

# make song usable to nn	
def transform_song(song, sr, time_length):
	song = splitsong(song, sr, time_length)
	song = np.expand_dims(song, axis=3)
	return song

# make all songs in train/test data usable to nn	
def transform_songs(songs, genres, sr, time_length):
	print('transforming songs')
	x = []
	y = []
	for i in range(songs.shape[0]):
		song = transform_song(songs[i], sr, time_length)
		x.extend(song)
		y.extend(song.shape[0]*[genres[i]])
	print('finished transforming songs')
	return np.asarray(x), np.asarray(y)