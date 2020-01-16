import sys
sys.path.append('..')
from network.AEI_Net import *
import time

device = torch.device('cuda:0')
torch.backends.cudnn.benchmar = True

net = AEI_Net(256)
net.eval().to(device)

batch_size = 4

z_id = torch.ones([batch_size, 256]).to(device)
x = torch.zeros([batch_size, 3, 256, 256]).to(device)

while True:
    st = time.time()
    y = net(x, z_id)
    y.mean().backward()
    st = time.time() - st
    print(f'{st} sec')
