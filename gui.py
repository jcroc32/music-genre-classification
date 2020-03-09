import sys
import os
import librosa
import sounddevice
import numpy as np
from keras.models import load_model
from model.transform_data import transform_song, pad_song
from model.nn import predict_song
# make compatabile for python 2 and 3
if sys.version_info[0] < 3:
	from Tkinter import *
	from tkFileDialog import askopenfilename
else:
	from tkinter import *
	from tkinter.filedialog import askopenfilename
#what we resample song to
resample_sr = 22050
# time length of song segments
time_length = 3
global song
genres = np.loadtxt('model/data/genres.txt', dtype=str, delimiter=" ")
model = load_model('model/model.h5')
# function for loading song
def get_song():
	genre_output.delete(0.0, END)
	app_status.delete(0.0, END)
	file_name = askopenfilename(title='Open song file') 
	try:
		app_status.insert('insert', 'getting song...\n')
		app_status.update()
		global song
		song, sr = librosa.load(file_name, sr=resample_sr, mono=True, offset=0.0, duration=None)
		app_status.insert('insert', 'sucessfully got '+os.path.basename(file_name)+'!\n')
		app_status.update()
		sounddevice.play(song, resample_sr)
	except Exception as e:
		print(e.args)
		app_status.insert('insert', 'format not recognized or file not found\n'
		'try opening another file\n(must be an audio file)\n')
		app_status.update()
# function for finding song's genre
def predict_genre():
	genre_output.delete(0.0, END)
	genre_output.insert('insert', 'predicting genre...\n')
	genre_output.update()
	try:
		padded_song = pad_song(song, resample_sr, time_length)
		genre = predict_song(padded_song, genres, transform_song, resample_sr, time_length, model)
		textvar = "The song's genre is: %s!" %(genre)
		genre_output.insert('insert', textvar+'\n')
		genre_output.update()
	except Exception as e:
		print(e.args)
		# couldn't find test.wav or model.h5 most likely
		genre_output.insert('insert', 'no loaded song found\n'
		'try opening another file\n')
		genre_output.update()
top = Tk()
top.title('Genre Classifier')
top.geometry('600x250')
canvas = Canvas(top, width=160, height=160, bd=0, bg='white')
canvas.grid(row=1, column=0)
get_song_button = Button(top, text='Open/Play', command=get_song)
get_song_button.grid(row=2, column=0)
predict_genre_button = Button(top, text ='Predict Genre', command=predict_genre)
predict_genre_button.grid(row=2, column=1)
stop_song_button = Button(top, text='Stop Music', command=sounddevice.stop)
stop_song_button.grid(row=3, column=0)
instructions = Label(top, text='Please <Open/Play> a song file, then press <Predict Genre> ')
instructions.grid(row=0)
app_status = Text(top, bd=0, width=32, height=10, font='Fixdsys -14')
app_status.grid(row=1, column=0)
genre_output = Text(top, bd=0, width=32, height=10, font='Fixdsys -14')
genre_output.grid(row=1, column=1)
# keep application going till user closes it
top.mainloop()