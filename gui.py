import sys
import os
# figure out if using python 2 or 3
if sys.version_info[0] < 3:
	from Tkinter import *
	from tkFileDialog import askopenfilename
else:
	from tkinter import *
	from tkinter.filedialog import askopenfilename
import librosa
from keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy import stats
from model.transform_data import transform_song
# function for loading song
def getSong():
	text2.delete(0.0, END)
	text1.delete(0.0, END)
	file_name = askopenfilename(title='Open song file') 
	try:
		text1.insert('insert', 'getting song...\n')
		text1.update()
		wav, sr = librosa.core.load(file_name, sr=want_sr, mono=True, offset=0.0, duration=30)
		# create data file our nn can interpret
		librosa.output.write_wav('test.wav', wav, sr, norm=True)
		# print that file read sucessfully to application
		text1.insert('insert', 'sucessfully got '+os.path.basename(file_name)+'!\n')
		text1.update()
		sd.play(wav,sr)
	except Exception as e:
		print(e.args)
		# file not read right
		text1.insert('insert', 'format not recognized or file not found\n'
		'try opening another file\n(must be an audio file)\n')
		text1.update()
# function for finding song's genre
def Predict():
	text2.delete(0.0, END)
	text2.insert('insert', 'predicting genre...\n')
	text2.update()
	try:
		wav, sr = librosa.core.load('test.wav', sr=want_sr, mono=True, offset=0.0, duration=30)
		if wav.shape[0] < 30*sr:
			wav = np.append(wav, np.zeros(30*sr - wav.shape[0])) # pad with zeros to get 30 sec
		wav = transform_song(wav)
		# load our saved model
		model=load_model('model/model.h5')
		# predict class
		genre=stats.mode(model.predict_classes(wav))[0]
		# print class to application
		textvar = "The song's genre is: %s!" %(label_name[int(genre)])
		text2.insert('insert', textvar+'\n')
		text2.update()
	except Exception as e:
		print(e.args)
		# couldn't find test.wav or model.h5 most likely
		text2.insert('insert', 'no loaded song found\n'
		'try opening another file\n'
		'(should see test.wav in folder\n after loading song)\n')
		text2.update()
    
want_sr = 22050 #what we resample song to
# class labels
label_name = np.loadtxt('model/data/genres.txt', dtype=str, delimiter=" ")
# start tkinter app
top = Tk()
top.title = 'Genre Classifier'
# make app size 600 by 250
top.geometry('600x250')
# create canvas we can put our buttons and output text on
canvas = Canvas(top, width=160,height=160, bd=0,bg='white')
canvas.grid(row=1, column=0)
# button for opening song file
submit_button = Button(top, text ='Open/Play', command = getSong)
submit_button.grid(row=2, column=0)
# button for predicting song genre
submit_button = Button(top, text ='Predict', command = Predict)
submit_button.grid(row=2, column=1)
# button for stop playing song
submit_button = Button(top, text ='Stop Music', command = sd.stop)
submit_button.grid(row=3, column=0)
# simple instruction for running app
label1=Label(top,text='Please <Open/Play> a song file, then press <Predict> ')
label1.grid(row=0)
# reading file output
text1=Text(top,bd=0, width=32,height=10,font='Fixdsys -14')
text1.grid(row=1, column=0)
# predicting genre output
text2=Text(top,bd=0, width=32,height=10,font='Fixdsys -14')
text2.grid(row=1, column=1)
# keep application going till user closes it
top.mainloop()