from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers

def genre_model(num_genres, input_shape):
  model = Sequential()
  model.add(BatchNormalization())
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 4)))
  model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(MaxPooling2D(pool_size=(2, 4)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
  model.add(Dropout(0.2))
  model.add(Dense(num_genres, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return(model)