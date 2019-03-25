import numpy as np
from sklearn.model_selection import train_test_split
from pandas import get_dummies
from load import load_data
from transform_data import transform_songs
from nn import genre_model

# what we resample song to
sr = 22050
# load data
songs, genres = load_data(sr)
# transform song into more usable data
songs, genres = transform_songs(songs, genres)
# break data into train,validate, and testing data
x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.25, random_state=50)
x_train_part, x_validate, y_train_part, y_validate = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25, random_state=50)
# one hot encode genres
y_train = get_dummies(y_train)
y_train_part = get_dummies(y_train_part)
y_validate = get_dummies(y_validate)
y_test = get_dummies(y_test)
# get number of genres and their positions
y_values = y_train.columns
num_genres = len(y_values)
np.savetxt('data/genres.txt', y_values, delimiter=" ", fmt="%s")
# get the data from pandas
y_train = y_train.values
y_train_part = y_train_part.values
y_validate = y_validate.values
y_test = y_test.values
# get model
model = genre_model(num_genres, x_train[0].shape)
try:
	# train
	model.fit(x_train_part, y_train_part, epochs=50, batch_size=32, validation_data=(x_validate,y_validate))
	# test
	loss, accuracy = model.evaluate(x_test, y_test)
	print('test accuracy:', accuracy)
	print('test loss:', loss)  
	model.save('model.h5')
except Exception as e:
	print(e.args)
	try:
		model.save_weights('temp_weights.h5')
		print('error occurred, saving weights to model/temp_weights.h5')
	except:
		print('model not defined, no weights saved')