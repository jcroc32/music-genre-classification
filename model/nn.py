from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
# from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D

def genre_model(num_genres, input_shape):
	'''
	model = Sequential()
	model.add(BatchNormalization())
	
	model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=input_shape))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(32, (3,3), strides=(2,2), activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64, (3,3), strides=(4,4), activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(128, (3,3), strides=(8,8), activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	model.add(Dense(num_genres, activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	'''
	model = Sequential()
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=4))
	model.add(Conv1D(filters=16, kernel_size=7, activation='relu', dilation_rate=1))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=32, kernel_size=7, activation='relu', dilation_rate=2))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=64, kernel_size=7, activation='relu', dilation_rate=4))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=64, kernel_size=7, activation='relu', dilation_rate=8))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=32, kernel_size=7, activation='relu', dilation_rate=16))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(num_genres, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return(model)