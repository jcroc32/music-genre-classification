from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D

def genre_model(num_genres, input_shape):
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
	return(model)