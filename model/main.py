import numpy as np
from sklearn.model_selection import train_test_split
from pandas import get_dummies
from load import load_data
from transform_data import change_to_frequency_data
from nn import genre_model

# what we resample song to
sr = 4000 #22050
# load data
songs, genres = load_data(sr)
# shift to frequency domain
songs = change_to_frequency_data(songs,sr)
# break data into train,validate, and testing data
x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.25)
x_train_part, x_validate, y_train_part, y_validate = train_test_split(x_train, y_train, stratify=y_train, test_size=0.3333)
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
# make ndim = 3 for x tensor
x_train = np.expand_dims(x_train, axis=3)
x_train_part = np.expand_dims(x_train_part, axis=3)
x_validate = np.expand_dims(x_validate, axis=3)
x_test = np.expand_dims(x_test, axis=3)
# get model
model = genre_model(num_genres, x_train.shape[1:])
try:
  # train
  model.fit(x_train_part, y_train_part, epochs=30, batch_size=64, validation_data=(x_validate,y_validate))
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