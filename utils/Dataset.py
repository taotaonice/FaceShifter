from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import os
import cv2


class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.3):
        datasets = []
        embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            with open(f'{data_path}/embed.pkl', 'rb') as f:
                embed = pickle.load(f)
                embeds.append(embed)
        self.datasets = datasets
        self.embeds = embeds
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
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
        embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = self.transforms(Image.fromarray(Xs))

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = self.transforms(Image.fromarray(Xt))
            same_person = 0
        else:
            Xt = Xs.clone()
            same_person = 1
        return Xs, Xt, embed, same_person

    def __len__(self):
        return sum(self.N)
