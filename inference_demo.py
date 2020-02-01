import sys
sys.path.append('./face_modules/')
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from network.AEI_Net import *
from face_modules.mtcnn import *
import cv2
import PIL.Image as Image
import numpy as np

detector = MTCNN()
device = torch.device('cuda')
G = AEI_Net(c_id=512)
G.eval()
G.load_state_dict(torch.load('./saved_models/G_latest.pth', map_location=torch.device('cpu')))
G = G.cuda()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

Xs_path = '/home/taotao/Pictures/u=3322705847,3022779128&fm=26&gp=0.jpg'
Xt_path = '/home/taotao/Pictures/u=3977885541,1855342996&fm=11&gp=0.jpg'

Xs_raw = cv2.imread(Xs_path)
Xt_raw = cv2.imread(Xt_path)
Xs = detector.align(Image.fromarray(Xs_raw[:, :, ::-1]), crop_size=(256, 256))
Xt = detector.align(Image.fromarray(Xt_raw[:, :, ::-1]), crop_size=(256, 256))

Xs_raw = np.array(Xs)[:, :, ::-1]
Xt_raw = np.array(Xt)[:, :, ::-1]

Xs = test_transform(Xs)
Xt = test_transform(Xt)

Xs = Xs.unsqueeze(0).cuda()
Xt = Xt.unsqueeze(0).cuda()
with torch.no_grad():
    embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
    embedt = arcface(F.interpolate(Xt[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
    Yt, _ = G(Xt, embeds)
    Ys, _ = G(Xs, embedt)
    Ys = Ys.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
    Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5

    Y = np.concatenate((Ys[:, :, ::-1], Yt[:, :, ::-1]), axis=1)
    X = np.concatenate((Xs_raw/255., Xt_raw/255.), axis=1)
    image = np.concatenate((X, Y), axis=0)
    cv2.imshow('image', image)
    cv2.waitKey(0)
