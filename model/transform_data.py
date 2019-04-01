import numpy as np
import librosa

def splitsong(song, window = 0.1, overlap = 0.5):
	x = []
	xshape = song.shape[0]
	chunk = int(xshape*window)
	offset = int(chunk*(1.-overlap))
	for s in [song[i:i+chunk] for i in range(0, xshape-chunk+offset, offset)]:
		x.append(s)
	return np.asarray(x)
	
def to_melspectrogram(song, n_fft = 1024, hop_length = 512):
	x = []
	for i in range(song.shape[0]):
		x.append(librosa.feature.melspectrogram(song[i], n_fft = n_fft, hop_length = hop_length))
	return np.asarray(x)
	
def transform_song(song):
	song = splitsong(song)
	#song = to_melspectrogram(song)
	#x = x.reshape((x.shape[1],x.shape[2],x.shape[0]))
	song = np.expand_dims(song, axis=3)
	return song
	
def transform_songs(songs, genres):
	print('transforming songs')
	x = []
	y = []
	for i in range(songs.shape[0]):
		song = transform_song(songs[i])
		x.extend(song)
		y.extend(song.shape[0]*[genres[i]])
	print('finished transforming songs')
	return np.asarray(x), np.asarray(y)