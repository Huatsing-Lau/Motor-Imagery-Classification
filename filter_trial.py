from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def filter_bandpass(data):
    b, a = signal.butter(7, [0.00625, 0.35], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData
#plt.plot(x,filtedData)
#plt.show()
