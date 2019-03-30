# 636-project
nn for genre classification  
to run final product: python gui.python  
to train your own model: cd model && python main.py  
File structure of dataset should be:  
dataset  
 - genre1  
   . genre1file1.wav  
   . genre1file2.wav  
   ...  
 -genre2  
   . genre2file1.wav  
   ...  
Obviously the names of the files and folders will be different, but all songs should be grouped into a folder with
the songs' genre name; all of the genre folders should be in one main folder. When training is done, direct the 
file explorer to this main folder and the program will get all files from the main folder's subdirectories (the 
genre folders). The songs do not need to be .wav files, any common audio file should do. The network only trains on 
the first 30 seconds of the song and will pad songs with silence if they are shorter than 30 seconds.