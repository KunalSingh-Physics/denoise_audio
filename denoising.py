import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plot
import numpy.fft
import librosa.display


# read file
print('loading.....')
file    = r'noiseaudio.wav'
sig, fs = librosa.load(file, sr=50000)

# Filtering
dt=1/fs
t=np.arange(0,6,dt)
n=len(t)

fhat = np.fft.fft(sig,n)                     # Compute the FFT
PSD = fhat * np.conj(fhat) / n             # Power spectrum (power per freq)
freq =(1/(dt*n)) * np.arange(n)           # Create x-axis of frequencies in Hz
L = np.arange(1,np.floor(n/20),dtype='int')
indices = PSD > 0.1 # Find all freqs with large power
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat     # Zero out small Fourier coeffs. 
ffilt = np.fft.ifft(fhat) 
plot.xlabel('frequency')
plot.ylabel('power')
plot.plot(freq[L],PSD.real[L],color='r',LineWidth=2,label='Noisy')
plot.xticks(range(int(freq[L[0]]),int(freq[L[-1]]),250))
plot.legend()
plot.show()
plot.figure()
plot.plot(freq[L],PSDclean[L],color='b',LineWidth=1.5,label='Filtered')
plot.xticks(range(int(freq[L[0]]),int(freq[L[-1]]),250))
plot.xlabel('frequency')
plot.ylabel('power')
plot.legend()
plot.show()


nsig=ffilt.real #converting to real
#spectrogram conversion 

abs_spectrogram = np.abs(librosa.stft(nsig))
plot.figure()
audio_signal= librosa.griffinlim(abs_spectrogram)
powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(audio_signal,fs)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()     

##reconstruct new audio
#sf.write('cleanaudio.wav', audio_signal, fs)




