import sys
sys.path.append('..')
from network.AEI_Net import *
import torch.optim as opt
import time
from apex import amp

device = torch.device('cuda:0')
print(torch.backends.cudnn.benchmark)
torch.backends.cudnn.benchmark = True

net = AEI_Net(256).cuda()
optm = opt.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
net, optm = amp.initialize(net, optm, opt_level="O3")

batch_size = 2

z_id = torch.ones([batch_size, 256]).to(device)
x = torch.zeros([batch_size, 3, 256, 256]).to(device)

while True:
    st = time.time()
    y = net(x, z_id)
    loss = y.mean()
    with amp.scale_loss(loss, optm) as scaled_loss:
        scaled_loss.backward()
    st = time.time() - st
    print(f'{st} sec')
