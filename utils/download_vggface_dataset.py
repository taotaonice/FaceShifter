#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import os
import threading
import socket
import urllib.request

timeout = 1
socket.setdefaulttimeout(timeout)

save_path = '/media/taotao/2T/vgg_face_dataset/'


def download_and_save(url, savename):
    try:
        data = urllib.request.urlopen(url).read()
        fid = open(savename, 'w+b')
        fid.write(data)
        print("download succeed: " + url)
        fid.close()
    except IOError:
        print("download failed: " + url)


def get_all_iamge(filename):
    fid = open(filename)
    name = filename.split('/')[-1]
    name = name[:-4]
    lines = fid.readlines()
    fid.close()
    for line in lines:
        line_split = line.split(' ')
        image_id = line_split[0]
        image_url = line_split[1]
        if not os.path.exists(f'{save_path}/' + name):
            os.mkdir(f'{save_path}/' + name)
        savefile = f'{save_path}/' + name + '/' + image_id + '.jpg'
        # The maxSize of Thread numberr:1000
        print(image_url, savefile)
        while True:
            if (len(threading.enumerate()) < 1000):
                break
        t = threading.Thread(target=download_and_save, args=(image_url, savefile,))
        t.start()


if __name__ == "__main__":
    fileDir = '/home/taotao/Downloads/vgg_face_dataset/files/'
    names = os.listdir(fileDir)
    for i in range(len(names)):
        get_all_iamge(os.path.join(fileDir, names[i]))
