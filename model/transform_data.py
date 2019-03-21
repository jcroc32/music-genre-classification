import numpy as np
import librosa

def change_song_to_frequency_data(X, sr = 22050):
  n_fft = 512
  hop_length = int(n_fft/2)
  n_mels = 64
  fmax = sr/2
  return librosa.feature.melspectrogram(X, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
  
def change_to_frequency_data(X, sr = 22050):
  data = []
  for i in range(X.shape[0]):
      data.append( change_song_to_frequency_data(X[i], sr = 22050) )
  return np.asarray(data)