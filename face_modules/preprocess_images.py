import torch
from mtcnn import MTCNN
import cv2
import numpy as np

import PIL.Image as Image
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import os
import libnvjpeg
import pickle

img_root_dir = '/media/taotao/2T/ffhq/images1024x1024/'
save_path = '/media/taotao/2T/ffhq/ffhq_aligned_256x256/'
# embed_path = '/home/taotao/Downloads/celeb-aligned-256/embed.pkl'

device = torch.device('cuda:0')
mtcnn = MTCNN()

model = Backbone(50, 0.6, 'ir_se').to(device)
model.eval()
model.load_state_dict(torch.load('./model_ir_se50.pth'))

# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

decoder = libnvjpeg.py_NVJpegDecoder()

ind = 0
embed_map = {}

for root, dirs, files in os.walk(img_root_dir):
    for name in files:
        if name.endswith('jpg') or name.endswith('png'):
            try:
                p = os.path.join(root, name)
                img = cv2.imread(p)
                face = mtcnn.align(Image.fromarray(img[:, :, ::-1]), crop_size=(256, 256))
                if face is None:
                    continue
                # scaled_img = face.resize((112, 112), Image.ANTIALIAS)
                # with torch.no_grad():
                #     embed = model(test_transform(scaled_img).unsqueeze(0).cuda()).squeeze().cpu().numpy()
                new_path = '%08d.jpg'%ind
                ind += 1
                print(new_path)
                face.save(os.path.join(save_path, new_path))
                # embed_map[new_path] = embed.detach().cpu()
            except Exception as e:
                continue

# with open(embed_path, 'wb') as f:
#     pickle.dump(embed_map, f)
#
# img = cv2.imread('/home/taotao/Pictures/47d947b4d9cf3e2f62c0c8023a1c0dea.jpg')[:,:,::-1]
# # bboxes, faces = mtcnn.align_multi(Image.fromarray(img), limit=10, min_face_size=30)
# bboxes, faces = mtcnn.align(Image.fromarray(img))
# input = test_transform(faces[0]).unsqueeze(0)
# embed = model(input.cuda())
# print(embed.shape)
# print(bboxes)
# face = np.array(faces[0])[:,:,::-1]
# cv2.imshow('', face)
# cv2.waitKey(0)
