import torch
import torchvision.transforms as T

con = torch.rand(500)
print('con:\n', con)
sub = torch.ones(502, 502)
print("sub:\n", sub)
sub[1:-1, 1:-1] = con
print('combined\n', sub.size())
blur = T.Compose([T.GaussianBlur(3, sigma=(0.1, 2.0))])
blrd = blur(sub[None, :, :])
print('blured:\n', blrd.size())

test = torch.rand(800, 800)[None, :, :]
arch = torch.empty(32, 800, 800)
print(arch.size())
for i in range(32):
	blur = T.Compose([T.GaussianBlur(2*i + 1, sigma=0.8)])
	blrd = blur(test)
	arch[i] = blrd[0]

print(arch[0], arch[0].size())
print(arch[12], arch[12].size())
print(arch[31], arch[31].size())
print(test[0])

