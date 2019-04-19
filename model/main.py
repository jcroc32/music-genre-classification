import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import get_dummies
from load import load_data
from transform_data import transform_song, transform_songs
from nn import genre_model, test_model

## parameters:
# location of this file (main.py)
file_path = os.path.abspath(os.path.dirname(__file__))
# file giving the order of genres (used to make sense of nn output)
genre_file = file_path+'/data/genres.txt'
# where we save model
model_file = file_path+'/model.h5'
# where we save weights in case of error
weights_file = file_path+'/temp_weights.h5'
# what sample rate to resample songs to
sr = 22050
# time length to cut the song into
time_length = 3

## for hyperparameter tunig
# use validation set
validate = False
# number of epochs
epochs = 22
# batch size
batch_size = 32
## 

# load data
songs, genres = load_data(sr)
# break data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.2, random_state=1)
# break into training and validation data
if(validate):
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=1)
	y_val = get_dummies(y_val)
	y_val = y_val.values
	x_val, y_val = transform_songs(x_val, y_val, sr, time_length)
	validation_data = (x_val,y_val)
	x_val = 0
	y_val = 0
else:
	validation_data = None
# clear some memory
songs = 0
genres = 0
# one hot encode genres
y_train = get_dummies(y_train)
y_test_transform = get_dummies(y_test)
# get number of genres and save genre order to file
genres = y_train.columns
num_genres = len(genres)
np.savetxt(genre_file, genres, delimiter=' ', fmt='%s')
# get the data from pandas
y_train = y_train.values
y_test_transform = y_test_transform.values
# transform song into more usable data
x_train, y_train = transform_songs(x_train, y_train, sr, time_length)
x_test_transform, y_test_transform = transform_songs(x_test, y_test_transform, sr, time_length)
# shuffle training data to mix genres
np.random.seed(1)
p = np.random.permutation(range(y_train.shape[0]))
x_train = x_train[p]
y_train = y_train[p]
# get model
model = genre_model(num_genres, x_train[0].shape)
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model 
try:
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
	historyDict = history.history
	# plot the accuracy and loss of training and validation data
	for key in historyDict.keys():
		plt.plot(historyDict[key],'.-')
	plt.legend(historyDict.keys())
	# save model to file
	model.save(model_file)
	# show plots
	plt.show()
# deal with errors
except Exception as e:
	print(e.args)
	try:
		model.save_weights(weights_file)
		print('error occurred, saving weights to '+weights_file)
	except:
		print('model not defined, no weights saved')
# test model on 3s interval
loss, accuracy = model.evaluate(x_test_transform, y_test_transform)
print('test 3s intervals accuracy:', accuracy)
print('test 3s intervals loss:', loss)
# test model on whole songs
test_model(x_test,y_test,genres,transform_song,sr,time_length,model)