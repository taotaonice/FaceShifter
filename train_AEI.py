from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
# from apex import amp
import visdom


vis = visdom.Visdom(server='49.235.201.74', env='faceshifter', port=8097)
batch_size = 16
lr_G = 1e-4
lr_D = 1e-3
max_epoch = 2000
show_step = 10
save_epoch = 1
model_save_path = './saved_models/'
optim_level = 'O2'

device = torch.device('cuda')
# torch.set_num_threads(12)

G = AEI_Net(c_id=512).to(device)
D = MultiscaleDiscriminator(input_nc=3, norm_layer=torch.nn.InstanceNorm2d).to(device)

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device))

opt_G = optim.Adam(G.parameters(), lr=lr_G, weight_decay=1e-4)
opt_D = optim.Adam(D.parameters(), lr=lr_D, weight_decay=1e-4)
try:
    G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')))
    D.load_state_dict(torch.load('./saved_models/D_latest.pth', map_location=torch.device('cpu')))
except Exception as e:
    print(e)
#G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
#D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

dataset = FaceEmbed(['../celeb-aligned-256/'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def get_numpy_image(X):
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=batch_size).numpy() * 0.5 + 0.5
    X = X.transpose([1,2,0])*255
    return X


def make_image(Xs, Xt, Y):
    Xs = get_numpy_image(Xs)
    Xt = get_numpy_image(Xt)
    Y = get_numpy_image(Y)
    return np.concatenate((Xs, Xt, Y), axis=0).transpose([2, 0, 1])


print(torch.backends.cudnn.benchmark)
torch.backends.cudnn.benchmark = True
for epoch in range(0, max_epoch):
    # torch.cuda.empty_cache()
    for iteration, data in enumerate(dataloader):
        print(f'{iteration} / {len(dataloader)}')
        start_time = time.time()
        Xs, Xt, embed, same_person = data
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        embed = embed.to(device)
        same_person = same_person.to(device)

        # train G
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        score = D(Y)[-1][0]
        L_adv = L1(score, torch.ones_like(score))

        ZY = arcface(F.interpolate(Y, [112, 112], mode='bilinear', align_corners=True))
        L_id = (1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += MSE(Xt_attr[i], Y_attr[i])
        L_attr /= 2.0

        L_rec = torch.mean(0.5 * torch.pow(Y - Xt, 2).reshape(batch_size, -1).mean(dim=1) * same_person)

        lossG = L_adv + 10*L_attr + 5*L_id + 10*L_rec
        # with amp.scale_loss(lossG, opt_G) as scaled_loss:
            # scaled_loss.backward()

        lossG.backward()
        opt_G.step()

        # train D
        opt_D.zero_grad()
        fake_score = D(Y.detach())[-1][0]
        true_score1 = D(Xs)[-1][0]
        true_score2 = D(Xt)[-1][0]

        lossD = 0.5*(L1(torch.zeros_like(fake_score), fake_score) +
                     0.5*(L1(true_score1, torch.ones_like(true_score1))
                          + L1(true_score2, torch.ones_like(true_score2))))

        # with amp.scale_loss(lossD, opt_D) as scaled_loss:
            # scaled_loss.backward()
        lossD.backward()
        opt_D.step()
        batch_time = time.time() - start_time
        if iteration % show_step == 0:
            image = make_image(Xs, Xt, Y)
            vis.image(image, opts={'title': 'result'}, win='result')
            cv2.imwrite('./gen_images/latest.jpg', image.transpose([1,2,0])[:,:,::-1])

        print(f'lossD: {lossD.item()}    lossG: {lossG.item()} batch_time: {batch_time}s')
    torch.save(G.state_dict(), './saved_models/G_latest.pth')
    torch.save(D.state_dict(), './saved_models/D_latest.pth')


