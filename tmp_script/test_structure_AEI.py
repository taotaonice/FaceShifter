import sys
sys.path.append('..')
from network.AEI_Net import *
from network.HEAR_Net import *
from network.MultiscaleDiscriminator import *
import torch.optim as opt
import time
from apex import amp

device = torch.device('cuda:0')
print(torch.backends.cudnn.benchmark)
torch.backends.cudnn.benchmark = True
#
# c_dim = 512
# net = AEI_Net(c_dim).cuda()
# D = MultiscaleDiscriminator(3)
# optm = opt.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# net, optm = amp.initialize(net, optm, opt_level="O2")
#
# batch_size = 2
#
# z_id = torch.ones([batch_size, c_dim]).to(device)
# x = torch.zeros([batch_size, 3, 256, 256]).to(device)
#
# while True:
#     st = time.time()
#     y = net(x, z_id)[0]
#     loss = y.mean()
#     with amp.scale_loss(loss, optm) as scaled_loss:
#         scaled_loss.backward()
#     attr = net.get_attr(x.half())
#     st = time.time() - st
#     print(f'{st} sec')

hearnet = HearNet()
hearnet.to(device)
hearnet.eval()
batch_size = 1

input = torch.zeros([batch_size, 6, 256, 256]).to(device)

with torch.no_grad():
    while True:
        st = time.time()
        Yst = hearnet(input)
        # Yst.mean().backward()
        st = time.time() - st
        print(st)
