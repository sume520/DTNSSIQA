import torch.utils.data as data
from torchvision import transforms

import folder


class DataLoader(object):
    def __init__(self, index, dataset, batch_size, status, num_worker):

        self.batch_size = batch_size
        self.dataset = dataset
        self.index = index
        self.statue = status
        self.num_worker = num_worker

        if dataset == 'palm':
            self.path = 'G:/IQADatabase/PalmIQA/data'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

            self.data = folder.PalmSet(root=self.path, transform=transform)

        elif dataset == 'palm2':
            self.path = 'G:/IQADatabase/PalmIQA/images'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

            self.data = folder.PalmSet2(root=self.path, index=self.index, transform=transform)

        elif dataset == 'livec':
            self.path = 'G:/IQADatabase/live_challenge'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

            self.data = folder.LiveCallengeSet(root=self.path, transform=transform)

        elif dataset == 'koniq-10k':
            self.path = 'G:/IQADatabase/kadid10k'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
            self.data = folder.Koniq_10kSet(root=self.path, transform=transform)

        elif dataset == 'csiq':
            self.path = 'G:/IQADatabase/csiq'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
            self.data = folder.CSIQSet(root=self.path, transform=transform)

        elif dataset == 'live':
            self.path = 'G:/IQADatabase/live'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
            self.data = folder.LiveSet(root=self.path, index=self.index, transform=transform)

        elif dataset == 'tid2013':
            self.path = 'G:/IQADatabase/tid2013'
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])
            self.data = folder.TID2013Set(root=self.path, transform=transform)

    def get_loader(self):

        if self.statue == 'train':
            data_loader = data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        else:
            data_loader = data.DataLoader(
                self.data, batch_size=1, shuffle=False, num_workers=self.num_worker)
        return data_loader
