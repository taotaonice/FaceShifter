from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import numpy as np
import os
import cv2


class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)


# Deprecated
class With_Identity(TensorDataset):
    def __init__(self, root_path, same_prob=0.8):
        self.root_path = root_path
        self.same_prob = same_prob
        self.classes = os.listdir(root_path)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        class_path = os.path.join(self.root_path, self.classes[item])
        files = glob.glob(class_path + '/*.*g')
        N = len(files)
        order = [i for i in range(N)]
        random.shuffle(order)
        Xs = Image.fromarray(cv2.imread(files[order[0]]))
        if random.random() < self.same_prob:
            if len(order) == 1:
                order.append(order[0])
            if random.random() < 0.5:
                order[1] = order[0]
            Xt = Image.fromarray(cv2.imread(files[order[1]]))
            return self.transforms(Xs), self.transforms(Xt), True
        else:
            other_class = random.randint(0, self.__len__()-1)
            class_path = os.path.join(self.root_path,
                                      self.classes[other_class])
            files = glob.glob(class_path + '/*.*g')
            pick = random.choice(files)
            Xt = Image.fromarray(cv2.imread(pick))
            return self.transforms(Xs), self.transforms(Xt), False

    def __len__(self):
        return len(self.classes)


def compose_occlusion(face_img, occlusions):
    h, w, c = face_img.shape
    if len(occlusions) == 0:
        return face_img
    for occlusion in occlusions:
        # scale
        scale = random.random() * 0.5 + 0.5
        # occlusion = cv2.resize(occlusion, (), fx=scale, fy=scale)
        # rotate
        R = cv2.getRotationMatrix2D((occlusion.shape[0]/2, occlusion.shape[1]/2), random.random()*180-90, scale)
        occlusion = cv2.warpAffine(occlusion, R, (occlusion.shape[1], occlusion.shape[0]))
        oh, ow, _ = occlusion.shape
        oc_color = occlusion[:, :, :3]
        oc_alpha = occlusion[:, :, 3].astype(np.float) / 255.
        oc_alpha = np.expand_dims(oc_alpha, axis=2)
        tmp = np.zeros([h+oh, w+ow, c])
        tmp[oh//2:oh//2+h, ow//2:ow//2+w, :] = face_img
        cx = random.randint(int(ow / 2) + 1, int(w + ow / 2) - 1)
        cy = random.randint(int(oh / 2) + 1, int(h + oh / 2) - 1)
        stx = cx - int(ow / 2)
        sty = cy - int(oh / 2)
        tmp[sty:sty+oh, stx:stx+ow, :] = oc_color * oc_alpha + tmp[sty:sty+oh, stx:stx+ow, :] * (1-oc_alpha)
        face_img = tmp[oh//2:oh//2+h, ow//2:ow//2+w, :].astype(np.uint8)
    return face_img


class AugmentedOcclusions(TensorDataset):
    def __init__(self, face_img_root, hand_sets, obj_sets, same_prob=0.5):
        self.same_prob = same_prob
        hands_data = []
        for hand_set_path in hand_sets:
            paths = glob.glob(hand_set_path + '/*.png')
            hands_data.extend(paths)
        self.hands_data = hands_data
        obj_data = []
        for obj_set_path in obj_sets:
            paths = glob.glob(obj_set_path + '/*.png')
            obj_data.extend(paths)
        self.obj_data = obj_data

        self.face_img_paths = glob.glob(face_img_root + '/*.jpg')
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def gen_occlusion(self):
        p = random.random()
        occlusions = []
        if p < 0.25: # no occlusion
            pass
        elif p < 0.5: # only hand
            hand_img = cv2.imread(self.hands_data[random.randint(0, len(self.hands_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(hand_img)
        elif p < 0.75: # only object
            obj_img = cv2.imread(self.obj_data[random.randint(0, len(self.obj_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(obj_img)
        else: # both
            hand_img = cv2.imread(self.hands_data[random.randint(0, len(self.hands_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(hand_img)
            obj_img = cv2.imread(self.obj_data[random.randint(0, len(self.obj_data)-1)], cv2.IMREAD_UNCHANGED)
            occlusions.append(obj_img)
        return occlusions

    def __getitem__(self, item):
        face_path = self.face_img_paths[item]
        face_img = cv2.imread(face_path)

        Xs = face_img
        p = random.random()
        if p > self.same_prob:
            Xt_path = self.face_img_paths[random.randint(0, len(self.face_img_paths)-1)]
            Xt = cv2.imread(Xt_path)
            Xt = compose_occlusion(Xt, self.gen_occlusion())
            same_person = 0
        else:
            Xt = compose_occlusion(face_img, self.gen_occlusion())
            same_person = 1
        return self.transforms(Image.fromarray(Xs)), self.transforms(Image.fromarray(Xt)), same_person

    def __len__(self):
        return len(self.face_img_paths)
