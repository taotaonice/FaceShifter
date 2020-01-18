from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch


batch_size = 2
lr_G = 1e-4
lr_D = 1e-4
max_epoch = 2000


G = AEI_Net(c_id=512).cuda()
D = MultiscaleDiscriminator(input_nc=3, norm_layer=torch.nn.InstanceNorm2d).cuda(1)

arcface = Backbone(50, 0.6, 'ir_se').cuda(1)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth'))

opt_G = optim.Adam(G.parameters(), lr=lr_G, weight_decay=1e-4)
opt_D = optim.Adam(D.parameters(), lr=lr_D, weight_decay=1e-4)

dataset = FaceEmbed(['/home/taotao/Downloads/celeb-aligned-256/'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()

for epoch in range(max_epoch):
    torch.cuda.empty_cache()
    for iteration, data in enumerate(dataloader):
        Xs, Xt, embed, same_person = data
        Xs = Xs.cuda()
        Xt = Xt.cuda()
        embed = embed.cuda()
        same_person = same_person.cuda()

        # train G
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        score = D(Y.cuda(1))[-1][0]
        L_adv = L1(score, torch.ones_like(score))

        ZY = arcface(F.interpolate(Y.cuda(1), [112, 112], mode='bilinear', align_corners=True))
        L_id = 1 - torch.cosine_similarity(embed, ZY.cuda(0), dim=1).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += MSE(Xt_attr[i], Y_attr[i])
        L_attr /= 2.0

        L_rec = torch.mean(0.5 * torch.pow(Y - Xt, 2).reshape(batch_size, -1).mean(dim=1) * same_person)

        lossG = L_adv.cuda(0) + 10*L_attr + 5*L_id + 10*L_rec

        lossG.backward()
        opt_G.step()
        del L_adv, L_attr, L_id, L_rec, ZY, Xt_attr

        # train D
        opt_D.zero_grad()
        fake_score = D(Y.cuda(1).detach())[-1][0]
        true_score1 = D(Xs.cuda(1))[-1][0]
        true_score2 = D(Xt.cuda(1))[-1][0]

        lossD = 0.5*(L1(torch.zeros_like(fake_score), fake_score) +
                     0.5*(L1(true_score1, torch.ones_like(true_score1))
                          + L1(true_score2, torch.ones_like(true_score2))))

        lossD.backward()
        opt_D.step()
        print(f'lossD: {lossD.item()}    lossG: {lossG.item()}')
        del fake_score, true_score2, true_score1, lossD, lossG
        torch.cuda.empty_cache()
