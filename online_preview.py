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

from Xlib import display, X

use_cuda_postprocess = True
if use_cuda_postprocess:
    from cuda_postprocess import CudaPostprocess
    postprocesser = CudaPostprocess(256, 256)

class Screen_Capture:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.dsp = display.Display()
        self.root = self.dsp.screen().root
        self.actw = self.dsp.intern_atom('_NET_ACTIVE_WINDOW')
        self.ids = []

    def read_frame(self):
        # W = self.W
        # H = self.H
        id = self.root.get_full_property(self.actw, X.AnyPropertyType).value[0]
        if len(self.ids) == 0:
            self.ids.append(id)
            return np.zeros([1,1,3]).astype(np.uint8)
        elif len(self.ids) == 1:
            if id == self.ids[0]:
                return np.zeros([1,1,3]).astype(np.uint8)
            else:
                self.ids.append(id)
        elif len(self.ids) == 2:
            if id != self.ids[1]:
                self.ids[0] = self.ids[1]
                self.ids[1] = id
        id = self.ids[0]
        focus = self.dsp.create_resource_object('window', id)
        geo = focus.get_geometry()
        H = geo.height
        W = geo.width
        raw = focus.get_image(0, 0, W, H, X.ZPixmap, 0xffffffff)
        image = Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")
        return np.array(image)


screen_capture = Screen_Capture(1080, 960)

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

Xs_paths = jjy
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
    embeds, Xs_feats = arcface(F.interpolate(Xses[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
    embeds = embeds.mean(dim=0, keepdim=True)


files = glob.glob('./tmp/3/*.*g')
files.sort()
ind = 0

mask = np.zeros([256, 256], dtype=np.float)
for i in range(256):
    for j in range(256):
        dist = np.sqrt((i-128)**2 + (j-128)**2)/128
        dist = np.minimum(dist, 1)
        mask[i, j] = 1-dist
mask = cv2.dilate(mask, None, iterations=20)
# for file in files[0:]:
#     print(file)
#     Xt_path = file
#     Xt_raw = cv2.imread(Xt_path)
cv2.namedWindow('image')#, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('image', 0, 0)
while True:
    try:
        Xt_raw = screen_capture.read_frame()
    except:
        continue
    # try:
    Xt, trans_inv = detector.align_fully(Image.fromarray(Xt_raw), crop_size=(256, 256),
                                         return_trans_inv=True, ori=[0,3,1])
    # except Exception as e:
    #     print(e)
    #     print('skip one frame')
    #     cv2.imshow('image', Xt_raw)
    #     cv2.imwrite('./write/%06d.jpg'%ind, Xt_raw)
    #     ind += 1
    #     cv2.waitKey(1)
    #     continue

    if Xt is None:
        cv2.imshow('image', Xt_raw[:,:,::-1])
        # cv2.imwrite('./write/%06d.jpg'%ind, Xt_raw)
        ind += 1
        cv2.waitKey(1)
        print('skip one frame')
        continue

    # Xt_raw = np.array(Xt)[:, :, ::-1]
    # Xt_raw = Xt_raw.astype(np.float)/255.0

    Xt = test_transform(Xt)

    Xt = Xt.unsqueeze(0).cuda()
    with torch.no_grad():
        # embeds = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        # embedt = arcface(F.interpolate(Xt[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        st = time.time()
        Yt, _ = G(Xt, embeds)
        Yt = Yt.squeeze().detach().cpu().numpy()
        st = time.time() - st
        print(f'inference time: {st} sec')
        # Ys, _ = G(Xs, embedt)
        # Ys = Ys.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        if not use_cuda_postprocess:
            Yt = Yt.transpose([1, 2, 0])*0.5 + 0.5
            Yt = Yt[:, :, ::-1]
            Yt_trans_inv = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
            mask_ = cv2.warpAffine(mask,trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
            mask_ = np.expand_dims(mask_, 2)
            Yt_trans_inv = mask_*Yt_trans_inv + (1-mask_)*(Xt_raw[:,:,::-1].astype(np.float)/255.)
        else:
            trans_inv = np.concatenate((trans_inv, np.array([0,0,1]).reshape(1, 3)), axis=0)
            trans = np.linalg.inv(trans_inv)
            trans = trans[:2, :]
            Yt_trans_inv = postprocesser.restore(Yt.copy(), mask, trans.copy(), Xt_raw, np.size(Xt_raw, 0), np.size(Xt_raw, 1))

        merge = Yt_trans_inv

        cv2.imshow('image', merge)
        # cv2.imwrite('./write/%06d.jpg'%ind, merge*255)
        ind += 1
        cv2.waitKey(1)
