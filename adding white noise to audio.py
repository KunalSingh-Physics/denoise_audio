from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


rate, data = wavfile.read(r'original audio.wav')

plt.plot(data)
plt.show()
print('og',data)

noise = np.random.normal(0,500,data.shape)

newaudio = data + noise

plt.plot(newaudio, 'g')
plt.show()
print('new',newaudio)

wavfile.write("noiseaudio.wav",rate,newaudio.astype(np.int16))

