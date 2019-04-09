from __future__ import print_function
import os
import sys
# figure out if using python 2 or 3
if sys.version_info[0] < 3:
	from Tkinter import Tk
	from tkFileDialog import askdirectory
	def input(s):
		return raw_input(s)
else:
	from tkinter import Tk
	from tkinter.filedialog import askdirectory
import numpy as np
import librosa

sr_ = 22050
data_file = os.path.abspath(os.path.dirname(__file__))+'/data/data.npz'
genre_folder = os.path.abspath(os.path.dirname(__file__))+'/../genres'

if not os.path.isdir(genre_folder) and not os.path.isfile(data_file):
	url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz' 
	input_s = input('not finding data files, have you downloaded data from '+url+' (yes/no)? ')
	if input_s == 'n' or input_s == 'no':
		print('please download tar file from '+url+' and run program again once file is unpacked')
		quit()

def load_data(sr = sr_):
	# no numpy file named data, have to read in all files to create data tensor
	if not os.path.isfile(data_file):
		songs, genres = read_from_raw_files(sr)
	else: # data tensor found in /data/data.npz
		songs, genres = read_from_data_file(sr)
	return songs, genres
  
def read_from_raw_files(sr = sr_):
	# window explorer to find main data folder
	print('select folder from explorer')
	Tk().withdraw()
	data_folder = askdirectory()
	# error in finding folder
	if not os.path.isdir(data_folder):
		print('folder not found, program exiting')
		quit()
	else:
		genre_folders = os.listdir(data_folder)
	# initialize data arrays  
	songs = []
	genres = []
	# go through each folder and read in data from all files
	for sub_folder in genre_folders:
		genre_path = data_folder + '/' + sub_folder
		audio_files = os.listdir(genre_path)
		print('Reading in '+sub_folder+' songs: ', end='')
		sys.stdout.flush()
		# get each file
		for audio_name in audio_files:
			wav, sr = librosa.core.load(genre_path + '/' + audio_name, sr=sr, mono=True, offset=0.0, duration=30)
			if wav.shape[0] < 30*sr:
				wav = np.append(wav, np.zeros(30*sr - wav.shape[0])) # pad with zeros to get 30s
			print('.', end='')
			sys.stdout.flush()
			# add data to array
			songs.append(wav)
			genres.append(sub_folder)
		print('\nfinished')
		sys.stdout.flush()
	songs = np.asarray(songs)
	genres = np.asarray(genres, dtype=str)
	print('saving data to /data/data.npz (this may take a while)')
	# save data arrays to file (/data/data.npz)
	np.savez(data_file, songs=songs, genres=genres, sr=sr)
	print('save finished')
	return songs, genres
  
def read_from_data_file(sr = sr_):
	print('reading in data from /data/data.npz (this may take a while)')
	data = np.load(data_file)
	songs = data['songs']
	genres = data['genres']
	if sr != data['sr']:
		input_string = input('sr found in /data.npz does not match the sr given ('
			+str(sr)+' != '+str(data['sr'])+') \nDo you want to re-read files? '
			+'(this may take a while and should only be done if absolutely necessary) ')
		if input_string == 'y' or input_string == 'yes':
			str_sr = str(data['sr'])
			data.close()
			os.rename(data_file,data_file[:-4]+str_sr+'.npz')
			return read_from_raw_files(sr)   
	print('finished reading data')
	return songs, genres