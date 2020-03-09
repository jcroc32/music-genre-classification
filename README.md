# Neural Network for genre classification  
To run final product: python gui.python  
To train your own model: cd model && python main.py  
File structure of dataset should be:  
dataset  
 - genre1  
   . genre1file1.wav  
   . genre1file2.wav  
   ...  
 - genre2  
   . genre2file1.wav  
   ...  
    
Obviously the names of the files and folders will be different, but all songs should be grouped into a folder with 
the songs' genre name; all of the genre folders should be in one main folder. When training the network, direct the 
file explorer to this main folder and the program will get all files from the main folder's subdirectories (the 
genre folders). The songs do not need to be .wav files, any common audio file should do.  
My dataset download: http://opihi.cs.uvic.ca/sound/genres.tar.gz Once downloaded, extract folder and put in 
music-genre-classification folder.  
Note: system to train and test nn used 16GB of system ram and 8GB vram. Using a system with lesser spec may have 
performance drop or may not run at all. 
