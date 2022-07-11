import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

url = './test1.jpg'
image = Image.open(url)

convert = T.ToTensor()
ten = convert(image)

size = 256
blur_channels = 12

arch = torch.empty(blur_channels, size, size, 3)
#print(arch.size())
for i in range(blur_channels):
        #blur = T.Compose([T.Resize((size, size)), T.GaussianBlur(2*i + 1, sigma=16.0)])	
        blur = T.Compose([T.Resize((size, size)), T.GaussianBlur((15, 15), sigma=(1.4 ** i + 0.001))])
        blrd = blur(ten[None, :, :])[0].permute(1,2,0)
        #print(blrd.size())
        arch[i] = blrd

#plt.imshow(arch[0])
#plt.show()
#plt.imshow(arch[7])
#plt.show()

with open('heatmap.npy', 'rb') as arr:
    heatmap = np.load(arr)

def make_histogram(narr):
    _ = plt.hist(heatmap, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

def normalize(data, channels):
    channels  -= 1
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) + (1/(2*channels))) * channels // 1

rearr = normalize(heatmap, blur_channels)
print(rearr)
print(rearr.size)
print(np.max(rearr))

outp = torch.empty(size, size, 3)
for i in range(size):
    for j in range(size):
        outp[i][j] = arch[int(rearr[i][j])][i][j]

plt.imshow(outp)
plt.show()
plt.imshow(rearr)
plt.show()

