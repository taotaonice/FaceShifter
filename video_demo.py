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
import glob

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

Xs_path = '/home/taotao/Pictures/timg.jpeg'
Xs_raw = cv2.imread(Xs_path)
Xs = detector.align(Image.fromarray(Xs_raw[:, :, ::-1]), crop_size=(256, 256))
Xs_raw = np.array(Xs)[:, :, ::-1]
Xs = test_transform(Xs)
Xs = Xs.unsqueeze(0).cuda()

with torch.no_grad():
    embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))


files = glob.glob('./tmp/*.*g')
files.sort()
ind = 0
for file in files:
    Xt_path = file
    # Xt_path = '/home/taotao/Pictures/u=3977885541,1855342996&fm=11&gp=0.jpg'
    Xt_raw = cv2.imread(Xt_path)
    try:
        Xt, trans_inv = detector.align(Image.fromarray(Xt_raw[:, :, ::-1]), crop_size=(256, 256), return_trans_inv=True)
    except Exception as e:
        print('skip one frame')
        continue

    if Xt is None:
        continue

    # Xt_raw = np.array(Xt)[:, :, ::-1]
    Xt_raw = Xt_raw.astype(np.float)/255.0

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cuda()
    with torch.no_grad():
        # embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        # embedt = arcface(F.interpolate(Xt[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        Yt, _ = G(Xt, embeds)
        # Ys, _ = G(Xs, embedt)
        # Ys = Ys.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt[:, :, ::-1]
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask = (Yt_trans_inv > 0).astype(np.float)
        mask = cv2.GaussianBlur(mask, (11, 11), 3)
        mask = (mask > 0.95)
        Yt_trans_inv = mask*Yt_trans_inv + (1-mask)*Xt_raw

        merge = np.concatenate((Xt_raw, Yt_trans_inv), axis=1)

        cv2.imshow('image', merge)
        cv2.imwrite('./write/%06d.jpg'%ind, merge*255)
        ind += 1
        cv2.waitKey(1)
