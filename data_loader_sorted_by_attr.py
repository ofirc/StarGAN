import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import time

class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, batch_size, image_size, transform, mode):
        self.image_path = image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        # remove the comment below to make buckets preprocessing way faster.
        #self.lines = self.lines[:6000]
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}
        self.seed = 1234
        self.file_n = ''

        print ('Start preprocessing dataset..!')
        random.seed(self.seed)
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        self.celebA_train_preprocess()

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling

        '''Build file names and labels lists for test and train'''
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def celebA_train_preprocess(self):
        '''Sort the lists by attributes'''
        # TODO: Currently only for Training phase. Need to consider testing as well
        print(time.ctime())
        print ('Start CelebA preprocessing dataset..!')       
        attr_baskets = []
        self.train_mb = []

        idx2str = {0:'black_hair', 1:'not_black_hair', 
                   2:'blond_hair', 3:'not_blond_hair', 
                   4:'brown_hair', 5:'not_brown_hair',
                   6:'male', 7:'female',
                   8:'young', 9:'old'}

        str2idx = {'black_hair':0, 'not_black_hair':1, 
                   'blond_hair':2, 'not_blond_hair':3, 
                   'brown_hair':4, 'not_brown_hair':5,
                   'male':6, 'female':7,
                   'young':8, 'old':9}

        # attr_baskets maps from feature/attribute into a list of tuples.
        # each tuple contains: (filename, [attributes])
        # e.g. 'not_black_hair': [('001766.jpg', [0, 0, 0, 1, 1])]
        attr_baskets = {'black_hair':[], 'not_black_hair':[], 
                        'blond_hair':[], 'not_blond_hair':[], 
                        'brown_hair':[], 'not_brown_hair':[],
                        'male':[], 'female':[],
                        'young':[], 'old':[]}

        # Step 0: dump only selected features in self.preprocess()
        #         self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        #         so for each sample we know only these features.

        # Step 1: divide sample into groups sorted by the feature or absence thereof
        #         e.g. attr_baskets['black_hair'] - contains all training samples w/ black hair
        #              attr_baskets['not_black_hair'] - contains all training samples w/o black hair
        for filename, labels in zip(self.train_filenames, self.train_labels):
            for idx, label in enumerate(labels):
                # if labels[2] == 1 then flip_label == 0. So, we'll add the item to attr_baskets['brown_hair']
                flip_label = 1 - label
                attr_baskets[idx2str[2*idx + flip_label]].append((filename, labels))

        for basket_values in attr_baskets.values():
            self.seed += 30
            random.seed(self.seed)
            random.shuffle(basket_values)

        # like attr_baskets but now contains a list of 16-sized lists.
        minibatch_baskets = {'black_hair':[], 'not_black_hair':[], 
                             'blond_hair':[], 'not_blond_hair':[], 
                             'brown_hair':[], 'not_brown_hair':[],
                             'male':[], 'female':[],
                             'young':[], 'old':[]}
        
        keep_working = True
        from datetime import datetime
        start = datetime.now()
        iter=0

        # Step 2: split the attr baskets into minibatches
        while keep_working:
            '''In each loop we try to put a minibatch in every basket'''
            cur = datetime.now()
            print('keep_working1: iter=%d, elapsed=%s, time=%s' % (iter, cur - start, time.ctime()))
            iter += 1
            keep_working = False
            for b_key in attr_baskets:
                # Take a mini batch
                mb = []

                if len(attr_baskets[b_key]) > self.batch_size:
                    mb = attr_baskets[b_key][0:self.batch_size]  # mb stands for mini batch
                    attr_baskets[b_key] = attr_baskets[b_key][self.batch_size:]
                    minibatch_baskets[b_key].append(mb)
                    keep_working = True

                # Remove selcted mini batch items from all other lists so it won't be selected twice in one epoch
                for item in mb:
                    for b_list in attr_baskets.values():
                        try:
                            b_list.remove(item)
                        except:
                            pass

        keep_working = True
        iter=0
        prev = 0
        start = datetime.now()
        # Step 3: generate pairs of [real_attr, values]
        #         [0, ('001888.jpg', [1, 0, 0, 1, 1]), ('005426.jpg', [1, 0, 0, 0, 0]), 
        #         ... ('002899.jpg', [1, 0, 0, 0, 1])]
        # the real_attr is the index into the dictionary above.
        # it is needed so that we know what attribute to flip.
        while keep_working:
            cur = datetime.now()
            print('keep_working2: iter=%d, elapsed=%s, time=%s' % (iter, cur - start, time.ctime()))
            iter += 1
            keep_working = False
            for key in minibatch_baskets:
                if len(minibatch_baskets[key]) > 0:
                    # train_mb will contain minibatchs in the following format: 
                    # [common attribute location in labels, (filename_0, labels_0), ..., (filename_batch_size_minus_1, labels_batch_size_minus_1)]
                    self.train_mb.append([str2idx[key]] + minibatch_baskets[key][0])
                    minibatch_baskets[key] = minibatch_baskets[key][1:]
                    keep_working = True

        print ('Finished CelebA preprocessing dataset..!')       
        print(time.ctime())

    def __getitem__(self, index):
        if self.mode == 'train':
            mb = self.train_mb[index]
            attribute_idx = mb[0]
            images = torch.zeros(self.batch_size, 3, self.image_size, self.image_size)
            labels = torch.zeros(self.batch_size, len(self.selected_attrs))

            for i in range(self.batch_size):
                # mb[1] is a tuple of (filename_0, labels_0)
                filename = mb[i+1][0]
                label = mb[i+1][1]
                image = Image.open(os.path.join(self.image_path, filename))
                images[i] = self.transform(image)
                labels[i] = torch.FloatTensor(label)
            
            return attribute_idx, images, labels
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        #self.celebA_train_preprocess()
        if self.mode == 'train':
            return len(self.train_mb)
        elif self.mode in ['test']:
            return self.num_data


def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, metadata_path, batch_size, image_size, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_path, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)
    return data_loader
