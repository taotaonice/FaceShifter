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
import time

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

jjy = glob.glob('/home/taotao/jjy/*.*g')
yy = glob.glob('/home/taotao/yy/*.*g')
dlrb = glob.glob('/home/taotao/dlrb/*.*g')
wsq = ['/home/taotao/Pictures/_-2022699153__-1119499174_1580813720175_1580813720000_wifi_0_1580813837000.jpg', '/home/taotao/Pictures/1580813775628.jpeg']
ew = ['/home/taotao/Pictures/u=670719782,34416986&fm=26&gp=0.jpg', '/home/taotao/Pictures/u=1509480533,2094244881&fm=26&gp=0.jpg']
fj = ['/home/taotao/Pictures/u=2213259300,1999166607&fm=26&gp=0.jpg']
ty = ['/home/taotao/Pictures/u=2926637442,3350514777&fm=26&gp=0.jpg']
tly = ['/home/taotao/Pictures/u=3216638246,2194008022&fm=26&gp=0.jpg']
ft = ['/home/taotao/Pictures/b03533fa828ba61ebe7db556bb17ce0f314e59e4.png', '/home/taotao/Pictures/b999a9014c086e06eaeb811975825df20bd1cbb6.jpeg']
alt = ['/home/taotao/Pictures/20190224235052_8154c3a8b1961200d86bfc7b74edc0f4_2_mwpm_03200403.jpg']
ycy = ['/home/taotao/Pictures/u=1341915507,1137570584&fm=26&gp=0.jpg', '/home/taotao/Pictures/u=1608176690,2383619727&fm=26&gp=0.jpg', '/home/taotao/Pictures/u=2262852619,3494679591&fm=11&gp=0.jpg', '/home/taotao/Pictures/u=3194463926,4053253650&fm=26&gp=0.jpg', '/home/taotao/Pictures/u=3482871913,2597348063&fm=26&gp=0.jpg']
lax = glob.glob('/home/taotao/Pictures/Screenshot from 2020-02-06*.png')
wsq.append('/home/taotao/Pictures/201526204015.jpg')

Xs_paths = wsq
Xs_raws = [cv2.imread(Xs_path) for Xs_path in Xs_paths]
Xses = []
for Xs_raw in Xs_raws:
    try:
        Xs = detector.align(Image.fromarray(Xs_raw[:, :, ::-1]), crop_size=(256, 256))
        # Xs_raw = np.array(Xs)[:, :, ::-1]
        Xs = test_transform(Xs)
        Xs = Xs.unsqueeze(0).cuda()
        Xses.append(Xs)
    except:
        continue
Xses = torch.cat(Xses, dim=0)
with torch.no_grad():
    embeds = arcface(F.interpolate(Xses[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True)).mean(dim=0, keepdim=True)


files = glob.glob('./tmp/3/*.*g')
files.sort()
ind = 0

mask = np.zeros([256, 256], dtype=np.float)
for i in range(256):
    for j in range(256):
        dist = np.sqrt((i-128)**2 + (j-128)**2)/128
        dist = np.minimum(dist, 1)
        mask[i, j] = 1-dist
mask = cv2.dilate(mask, None, iterations=40)
for file in files[000:]:
    print(file)
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
        st = time.time()
        Yt, _ = G(Xt, embeds)
        st = time.time() - st
        print(f'inference time: {st} sec')
        # Ys, _ = G(Xs, embedt)
        # Ys = Ys.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        Yt = Yt[:, :, ::-1]
        Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask_ = np.expand_dims(mask_, 2)
        # mask = (Yt_trans_inv > 0).astype(np.float)
        # mask = cv2.GaussianBlur(mask, (11, 11), 3)
        # mask = (mask > 0.95)
        Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*Xt_raw

        # merge = np.concatenate((Xt_raw, Yt_trans_inv), axis=1)
        merge = Yt_trans_inv

        cv2.imshow('image', merge)
        cv2.imwrite('./write/%06d.jpg'%ind, merge*255)
        ind += 1
        cv2.waitKey(1)
