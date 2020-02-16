import sys
sys.path.append('../Peppa_Pig_Face_Engine-master/')
sys.path.append('../')
from lib.core.api.facer import FaceAna

from lib.core.headpose.pose import get_head_pose, line_pairs

facer = FaceAna()

import torch
import torchvision.transforms as transforms
from face_modules.model import Backbone
from network.AEI_Net import *
import cv2
import PIL.Image as Image
import numpy as np
from face_modules.mtcnn_pytorch.src.align_trans import *


device = torch.device('cuda')

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('../face_modules/model_ir_se50.pth', map_location=device), strict=False)

test_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_f5p(landmarks):
    eye_left = landmarks[36:41].mean(axis=0)
    eye_right = landmarks[42:47].mean(axis=0)
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]
    f5p = [[eye_left[0], eye_left[1]],
           [eye_right[0], eye_right[1]],
           [nose[0], nose[1]],
           [mouth_left[0], mouth_left[1]],
           [mouth_right[0], mouth_right[1]]]
    return f5p


dn0 = '/home/taotao/Pictures/7e3e6709c93d70cf8cb965a4f6dcd100bba12bdc.jpeg'
dn1 = '/home/taotao/Pictures/b3b7d0a20cf431ada7388e904536acaf2fdd98b5.jpg'
dn2 = '/home/taotao/Pictures/u=3226707260,1696055340&fm=26&gp=0.jpg'

ft0 = '/home/taotao/Pictures/9e3df8dcd100baa159770840d2289817c9fc2eab.jpeg'
ft1 = '/home/taotao/Pictures/b999a9014c086e06eaeb811975825df20bd1cbb6.jpeg'
ft2 = '/home/taotao/Pictures/b03533fa828ba61ebe7db556bb17ce0f314e59e4.png'

ew0 = '/home/taotao/Pictures/u=670719782,34416986&fm=26&gp=0.jpg'
ew1 = '/home/taotao/Pictures/u=1509480533,2094244881&fm=26&gp=0.jpg'

sjl0 = '/home/taotao/Pictures/asdgsdasf.jpeg'
sjl1 = '/home/taotao/Pictures/u=1912807554,30254209&fm=26&gp=0.jpg'
sjl2 = '/home/taotao/Pictures/u=3322705847,3022779128&fm=26&gp=0.jpg'

dlrb0 = '/home/taotao/dlrb/00013950.jpg'
dlrb1 = '/home/taotao/matlabspace/dlrb/000092.jpg'
dlrb2 = '/home/taotao/Pictures/Screenshot from 2020-02-15 14-46-32.png'

fj0 = '/home/taotao/fj/00105490.jpg'
fj1 = '/home/taotao/fj/00105566.jpg'
fj2 = '/home/taotao/fj/00105548.jpg'

A_path = dn0
B_path = ft0
faces = [cv2.imread(A_path), cv2.imread(B_path)]

emb = []
feats = []
for face in faces:
    Xs_raw = face
    _, landmarks, _ = facer.run(Xs_raw)
    f5p = get_f5p(landmarks[0])
    Xs = warp_and_crop_face(Xs_raw, f5p, reference_pts=get_reference_facial_points(default_square=True), crop_size=(256, 256))
    cv2.imshow("", Xs)
    cv2.waitKey(0)
    Xs = Image.fromarray(Xs)
    Xs = test_transform(Xs)
    Xs = Xs.unsqueeze(0).cuda()

    with torch.no_grad():
        embeds, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
    emb.append(embeds)
    feats.append(Xs_feats)

emba, embb = emb[0], emb[1]
# emba = emba.view(-1)
# embb = embb.view(-1)

print(f'embed norm diff: {(emba - embb).norm()}')
print(f'cosine similarity loss: {1-torch.cosine_similarity(emba, embb)}')

for i in range(len(feats[0])):
    fa = feats[0][i]
    fb = feats[1][i]

    fa = fa.view(-1)
    fb = fb.view(-1)

    print(f'layer {i} norm diff: {(fa-fb).norm()/fa.norm()} '
          f'mean abs diff: {torch.abs(fa-fb).mean()}')

