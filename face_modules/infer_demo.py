import torch
from mtcnn import MTCNN
import cv2
import numpy as np

import PIL.Image as Image
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans

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


img = cv2.imread('/home/taotao/Downloads/celeba-512/000014.jpg.jpg')[:, :, ::-1]

bboxes, faces = mtcnn.align_multi(Image.fromarray(img), limit=10, min_face_size=30)
input = test_transform(faces[0]).unsqueeze(0)
embbed = model(input.cuda())
print(embbed.shape)
print(bboxes)
face = np.array(faces[0])[:,:,::-1]
cv2.imshow('', face)
cv2.waitKey(0)
