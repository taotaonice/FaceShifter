from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
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
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)


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
        Xs = Image.fromarray(cv2.imread(files[order[0]])[:, :, ::-1])
        if random.random() < self.same_prob:
            if len(order) == 1:
                order.append(order[0])
            if random.random() < 0.5:
                order[1] = order[0]
            Xt = Image.fromarray(cv2.imread(files[order[1]])[:, :, ::-1])
            return self.transforms(Xs), self.transforms(Xt), True
        else:
            other_class = random.randint(0, self.__len__()-1)
            class_path = os.path.join(self.root_path,
                                      self.classes[other_class])
            files = glob.glob(class_path + '/*.*g')
            pick = random.choice(files)
            Xt = Image.fromarray(cv2.imread(pick)[:, :, ::-1])
            return self.transforms(Xs), self.transforms(Xt), False

    def __len__(self):
        return len(self.classes)
