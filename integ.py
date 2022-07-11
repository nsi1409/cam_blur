import numpy as np
import matplotlib.pyplot as plt
import cv2

with open('heatmap.npy', 'rb') as arr:
    heatmap = np.load(arr)

print(heatmap)

def make_histogram(narr):
    _ = plt.hist(heatmap, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

def normalize(data, channels):
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) + (1/(2*channels))) * channels // 1

rearr = normalize(heatmap, 32)
print(rearr)
print(rearr.size)
print(np.max(rearr))

plt.imshow(rearr)
plt.show()

