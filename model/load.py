from __future__ import print_function
import os
import sys
import librosa
import numpy as np
from transform_data import pad_song
# make compatabile for python 2 and 3
if sys.version_info[0] < 3:
	from Tkinter import Tk
	from tkFileDialog import askdirectory
	def input(s):
		return raw_input(s)
else:
	from tkinter import Tk
	from tkinter.filedialog import askdirectory
# where the dataset extracted from audio files is stored
data_file = os.path.abspath(os.path.dirname(__file__))+'/data/data.npz'
# possible location of raw audio files
genre_folder = os.path.abspath(os.path.dirname(__file__))+'/../genres'
# check if either of the 2 file/folder above exist
if not os.path.isdir(genre_folder) and not os.path.isfile(data_file):
	url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz' 
	input_s = input('\n Not finding data files, have you downloaded data from '+url+'? (yes/no) ')
	if input_s == 'n' or input_s == 'no':
		print('\n Please download tar file from '+url+' and run program again once file is unpacked')
		quit()
# general function to load dataset
def load_data(sr, time_length):
	if not os.path.isfile(data_file):
		songs, genres = read_from_raw_files(sr, time_length)
	input_s = input('\n Reload the data from source? (i.e. read from audio files, not cached .npz file: yes/no) ')
	if input_s != 'n' and input_s != 'no':
		songs, genres = read_from_raw_files(sr, time_length)
	else:
		songs, genres = read_from_data_file(sr)
	return songs, genres
# function called if we need to extract dataset from raw audio files
def read_from_raw_files(sr, time_length):
	if os.path.isdir(genre_folder):
		data_folder = genre_folder
	else:
		print('\n Select folder from explorer')
		Tk().withdraw()
		data_folder = askdirectory()
	if not os.path.isdir(data_folder):
		print('\n Folder not found, program exiting')
		quit()
	else:
		genre_folders = os.listdir(data_folder)
	songs = []
	genres = []
	for sub_folder in genre_folders:
		genre_path = data_folder + '/' + sub_folder
		audio_files = os.listdir(genre_path)
		print(' Reading in '+sub_folder+' songs: ', end='')
		sys.stdout.flush()
		for audio_name in audio_files:
			song, sr = librosa.load(genre_path + '/' + audio_name, sr=sr, mono=True, offset=0.0, duration=None)
			song = pad_song(song, sr, time_length)
			print('.', end='')
			sys.stdout.flush()
			songs.append(song)
			genres.append(sub_folder)
		print('\n Finished reading '+sub_folder+' songs')
		sys.stdout.flush()
	songs = np.array(songs)
	genres = np.array(genres, dtype=str)
	print('\n Saving data to ./data/data.npz (this may take a while)')
	np.savez(data_file, songs=songs, genres=genres, sr=sr)
	print(' Save finished')
	return songs, genres
# function called if we have dataset saved to a .npz file
def read_from_data_file(sr):
	print(' Reading in data from /data/data.npz (this may take a while)')
	data = np.load(data_file)
	songs = data['songs']
	genres = data['genres']
	if sr != data['sr']:
		input_string = input('\n Sr found in /data.npz does not match the sr given ('
			+str(sr)+' != '+str(data['sr'])+') \n Do you want to re-read raw files? '
			+'(this may take a while and is only necessary if the model is dramatically changed) ')
		if input_string == 'y' or input_string == 'yes':
			str_sr = str(data['sr'])
			data.close()
			os.rename(data_file,data_file[:-4]+str_sr+'.npz')
			return read_from_raw_files(sr)   
	print(' Finished reading data')
	return songs, genres