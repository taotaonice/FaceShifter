from network.AEI_Net import *
from network.HEAR_Net import *
from utils.Dataset import *
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import numpy as np
import torchvision
import cv2
from apex import amp
import visdom


vis = visdom.Visdom(server='127.0.0.1', env='faceshifter', port=8099)
batch_size = 32
lr = 4e-4
max_epoch = 2000
show_step = 10
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O1'

device = torch.device('cuda')

G = AEI_Net(c_id=512).to(device)
G.eval()
G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=True)

net = HearNet()
net.train()
net.to(device)

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

opt = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))

net, opt = amp.initialize(net, opt, opt_level=optim_level)

try:
    net.load_state_dict(torch.load('./saved_models/HEAR_latest.pth', map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print(e)

dataset = AugmentedOcclusions('../hearnet_data',
                              ['../ego_hands_png'],
                              ['../shapenet_png'], same_prob=0.5)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def get_numpy_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]).numpy() * 0.5 + 0.5
    X = X.transpose([1,2,0])*255
    np.clip(X, 0, 255).astype(np.uint8)
    return X


def make_image(Xs, Xt, Y):
    Xs = get_numpy_image(Xs)
    Xt = get_numpy_image(Xt)
    Y = get_numpy_image(Y)
    return np.concatenate((Xs, Xt, Y), axis=0).transpose([2, 0, 1])

print(torch.backends.cudnn.benchmark)
#torch.backends.cudnn.benchmark = True
for epoch in range(0, max_epoch):
    # torch.cuda.empty_cache()
    for iteration, data in enumerate(dataloader):
        start_time = time.time()
        Xs, Xt, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        with torch.no_grad():
            embed_s, _ = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            embed_t, _ = arcface(F.interpolate(Xt[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        same_person = same_person.to(device)

        # train G
        opt.zero_grad()
        with torch.no_grad():
            Yst_hat, _ = G(Xt, embed_s)
            Ytt, _ = G(Xt, embed_t)

        dYt = Xt - Ytt
        hear_input = torch.cat((Yst_hat, dYt), dim=1)
        Yst = net(hear_input)

        Yst_aligned = Yst[:, :, 19:237, 19:237]

        id_Yst, _ = arcface(F.interpolate(Yst_aligned, [112, 112], mode='bilinear', align_corners=True))

        L_id =(1 - torch.cosine_similarity(embed_s, id_Yst, dim=1)).mean()

        L_chg = L1(Yst_hat, Yst)

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Yst - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

        loss = L_id + L_chg + L_rec
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        opt.step()

        batch_time = time.time() - start_time
        if iteration % show_step == 0:
            image = make_image(Xs, Xt, Yst)
            vis.image(image, opts={'title': 'HEAR'}, win='HEAR')
            cv2.imwrite('./gen_images/HEAR_latest.jpg', image.transpose([1,2,0])[:,:,::-1])
        print(f'epoch: {epoch}    {iteration} / {len(dataloader)}')
        print(f'loss: {loss.item()} batch_time: {batch_time}s')
        print(f'L_id: {L_id.item()} L_chg: {L_chg.item()} L_rec: {L_rec.item()}')
        if iteration % 1000 == 0:
            torch.save(net.state_dict(), './saved_models/HEAR_latest.pth')


