import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from pandas import get_dummies

genre_file = os.path.abspath(os.path.dirname(__file__))+'/data/genres.txt'

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
	
def split_songs(songs, genres):
	# break data into train,validate, and testing data
	x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.25, random_state=50)
	x_train_part, x_validate, y_train_part, y_validate = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25, random_state=50)
	# one hot encode genres
	y_train_part = get_dummies(y_train_part)
	y_validate = get_dummies(y_validate)
	y_test = get_dummies(y_test)
	# get number of genres and their positions
	y_values = y_train_part.columns
	num_genres = len(y_values)
	np.savetxt(genre_file, y_values, delimiter=' ', fmt='%s')
	# get the data from pandas
	y_train_part = y_train_part.values
	y_validate = y_validate.values
	y_test = y_test.values
	return x_train_part, x_validate, x_test, y_train_part, y_validate, y_test, num_genres