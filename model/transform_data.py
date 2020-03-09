import numpy as np
# pad song so it splits up to equal length segments
def pad_song(song, sr, time_length):
	step = sr*time_length	
	padding_length = step - (len(song) % step)
	padding = np.array([0]*padding_length)
	song = np.append(song, padding)
	return song
# split song into segments of the given time length
def split_song(song, sr, time_length):
	step = sr*time_length
	num_rows = len(song) // step
	num_columns = step
	x = np.zeros((num_rows, num_columns))
	for i in range(num_rows):
		x[i,:] = song[i*step : (i+1)*step]
	return x
# transform a single song into data that the neural network can use
def transform_song(song, sr, time_length):
	song = split_song(song, sr, time_length)
	song = np.expand_dims(song, axis=2)
	return song
# transform all song
def transform_songs(songs, genres, sr, time_length):
	print(' Transforming some songs')
	x = []
	y = []
	for i in range(songs.shape[0]):
		song = transform_song(songs[i], sr, time_length)
		x.extend(song)
		y.extend(song.shape[0]*[genres[i]])
	print(' Finished transforming some songs')
	return np.array(x), np.array(y)