import csv
import os
import random

import numpy as np
import pandas as pd
import scipy.io
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import T_co


class PalmSet(data.Dataset):
    def __init__(self, root, transform):
        imagepath = os.listdir(root)
        imagepath.remove('labels.txt')

        imaganum = len(imagepath)
        labels = pd.read_csv(os.path.join(root, 'labels.txt')).to_numpy().astype(np.float32)

        sample = []
        for i in range(imaganum):
            sample.append((os.path.join(root, imagepath[i]), labels[i]))
        # 洗牌
        random.shuffle(sample)
        self.samples = sample
        self.transform = transform
        self.length = imaganum

    def __getitem__(self, index) -> T_co:
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        # length = len(self.samples)
        return self.length


class PalmSet2(data.Dataset):
    def __init__(self, root, index, transform):
        image_folders = os.listdir(root)
        image_folders.remove('targets.csv')
        imagepath = []
        for folder in image_folders:
            path = os.path.join(root, folder)
            for image_name in os.listdir(path):
                imagepath.append(os.path.join(path, image_name))

        labels = pd.read_csv(os.path.join(root, 'targets.csv'))
        labels = labels.set_index('ImageName')['Score'].to_dict()
        sample = []
        for i in range(len(index)):
            image_name = os.path.basename(imagepath[index[i]])
            sample.append((os.path.join(root, imagepath[index[i]]), np.float32(labels[image_name])))

        # 洗牌
        random.shuffle(sample)
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index) -> T_co:
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class LiveCallengeSet(data.Dataset):

    def __init__(self, root, transform):
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        # i 索引值 item index中
        for i in range(len(imgpath)):
            sample.append((os.path.join(root, 'Images', imgpath[i][0][0]), labels[i]))

        # 洗牌
        random.shuffle(sample)

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kSet(data.Dataset):

    def __init__(self, root, transform):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'dmos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['dist_img'])
                mos = np.array(float(row['dmos'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i in range(len(imgname)):
            sample.append((os.path.join(root, 'images', imgname[i]), mos_all[i]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LiveSet(data.Dataset):

    def __init__(self, root, index, transform):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos.mat'))
        labels = dmos['dmos'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []
        distort_type = 0
        for i in range(0, len(index)):
            image_sel = (refname[index[i]] == refnames_all)  # numpy数组通过与==与变量比较生成一个新的numpy数组，其中存储了numpy数组所有值与变量对比结果
            image_sel = image_sel * ~orgs.astype(np.bool_)  # 将train_sel中所有参考图像对应位置改成false
            image_sel = np.where(image_sel == True)  # where方法会根据维度返回多个数组，每个数组只包含同一维度坐标
            image_sel = image_sel[1].tolist()
            for j, item in enumerate(image_sel):
                if item in range(0, 227):
                    distort_type = 0
                if item in range(227, 460):
                    distort_type = 1
                if item in range(460, 634):
                    distort_type = 2
                if item in range(634, 808):
                    distort_type = 3
                if item in range(808, 982):
                    distort_type = 4
                sample.append((imgpath[item], labels[0][item], distort_type))

        self.samples = sample
        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, type = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, type


    def __len__(self):
        length = len(self.samples)
        return length


    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = 'img%d%s' % (index, '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LiveDistortedSet(data.Dataset):

    def __init__(self, root, index, transform):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        # 不同失真类型label范围为，jp2k:0-226,jpeg:227-449,wn:4500-623,gblur:624-797,fastfading:798-971
        dmos = scipy.io.loadmat(os.path.join(root, 'dmos.mat'))
        labels = dmos['dmos'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        # sample = []

        sample = {"je2k": [],
                  "jpeg": [],
                  "wn": [],
                  "gblur": [],
                  "fastfading": []}
        target = {"je2k": [],
                  "jpeg": [],
                  "wn": [],
                  "gblur": [],
                  "fastfading": []}

        for i in range(0, len(index)):
            image_sel = (refname[index[i]] == refnames_all)  # numpy数组通过与==与变量比较生成一个新的numpy数组，其中存储了numpy数组所有值与变量对比结果
            image_sel = image_sel * ~orgs.astype(np.bool_)  # 将train_sel中所有参考图像对应位置改成false
            image_sel = np.where(image_sel == True)  # where方法会根据维度返回多个数组，每个数组只包含同一维度坐标
            print(len(image_sel))
            image_sel = image_sel[1].tolist()
            for j, item in enumerate(image_sel):
                # sample.append((imgpath[item], labels[0][item]))
                if item in range(0, 227):
                    sample['je2k'].append(imgpath[item])
                    target['je2k'].append(imgpath[item])
                if item in range(227, 460):
                    sample['jpeg'].append(imgpath[item])
                    target['jpeg'].append(imgpath[item])
                if item in range(460, 634):
                    sample['wn'].append(imgpath[item])
                    target['wn'].append(imgpath[item])
                if item in range(634, 808):
                    sample['gblur'].append(imgpath[item])
                    target['gblur'].append(imgpath[item])
                if item in range(808, 982):
                    sample['fastfading'].append(imgpath[item])
                    target['fastfading'].append(imgpath[item])

        self.samples = [sample, target]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%03d%s' % (index, '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class CSIQSet(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))

        # 洗牌
        random.shuffle(sample)
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Set(data.Dataset):

    def __init__(self, root, transform):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        for i in range(len(imgnames)):
            sample.append((os.path.join(root, 'distorted_images', imgnames[i]), labels[i]))

        # 洗牌
        random.shuffle(sample)
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename
