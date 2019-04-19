import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from model.load import load_data
from model.nn import test_model
from model.transform_data import transform_song

sr = 22050
time_length = 3
songs, genres = load_data(sr)
x_train, x_test, y_train, y_test = train_test_split(songs, genres, stratify=genres, test_size=0.2, random_state=1)
genres = np.loadtxt('model/data/genres.txt', dtype=str, delimiter=" ")

model=load_model('model/model.h5')
model.summary()

test_model(x_test,y_test,genres,transform_song,sr,time_length,model)