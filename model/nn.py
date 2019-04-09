from keras import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D

def genre_model(num_genres, input_shape):
	input = Input(shape=input_shape)
	model = BatchNormalization()(input)
	model = Conv1D(filters=128, kernel_size=7, activation='relu', dilation_rate=1)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=64, kernel_size=7, activation='relu', dilation_rate=2)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=64, kernel_size=7, activation='relu', dilation_rate=4)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=32, kernel_size=7, activation='relu', dilation_rate=8)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=32, kernel_size=7, activation='relu', dilation_rate=16)(model)
	model = MaxPooling1D(pool_size=3)(model)
	model = Dropout(0.25)(model)
	model = Conv1D(filters=16, kernel_size=7, activation='relu', dilation_rate=32)(model)
	model = Flatten()(model)
	output = Dense(num_genres, activation='softmax')(model)
	model = Model(input,output)
	return(model)