import os
from load import load_data
from transform_data import transform_songs, split_songs
from nn import genre_model

file_path = os.path.abspath(os.path.dirname(__file__))

# what we resample song to
sr = 22050
# load data
songs, genres = load_data(sr)
# transform song into more usable data
songs, genres = transform_songs(songs, genres)
# split data into training, validating, and testing data
x_train_part, x_validate, x_test, y_train_part, y_validate, y_test, num_genres = split_songs(songs, genres)
# clear some memory
songs = 0
genres = 0
# get model
model = genre_model(num_genres, x_train_part[0].shape)
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
try:
	# train
	model.fit(x_train_part, y_train_part, epochs=50, batch_size=32, validation_data=(x_validate,y_validate))
	# test
	loss, accuracy = model.evaluate(x_test, y_test)
	print('test accuracy:', accuracy)
	print('test loss:', loss)
	model.save(file_path+'/model.h5')
except Exception as e:
	print(e.args)
	try:
		model.save_weights(file_path+'/temp_weights.h5')
		print('error occurred, saving weights to model/temp_weights.h5')
	except:
		print('model not defined, no weights saved')