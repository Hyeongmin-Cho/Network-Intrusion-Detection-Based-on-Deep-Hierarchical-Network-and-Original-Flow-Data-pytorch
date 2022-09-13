import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(directory, class_to_idx, extensions='.pkl'):
    instances = []
    directory = os.path.expanduser(directory)
    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances



class PklsFolder(Dataset): #CNN-LSTM 전처리를 위한
    def __init__(self, root_dir, dataset_name):
        classes, class_to_idx = self._find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.dataset_name = dataset_name

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort(key = lambda x : int(x))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = pickle.load(f)
        flow = np.zeros([10,160],dtype=np.uint8) #실험 1
        
        len_sample = len(sample)
        if(len_sample > 10) : len_sample = 10 #실험 1

        for i in range(len_sample): 
            flow[i, :len(sample[i][15:175])] = np.frombuffer(sample[i][15:175], dtype=np.uint8) 
        
        flow = flow.reshape(40,40) #Reshape the 1600-dimensional flow feature into a 40*40 grayscale image
        # 0 : Benign, 1 : Abnormal          
        if self.dataset_name == 'BoT':
            if target == 10:
                label = 0
            else:
                label = 1
        elif self.dataset_name == 'ToN':
            if target == 1:
                label = 0
            else:
                label = 1
        elif self.dataset_name == 'ISCX2012' or self.dataset_name == 'ISCX2017':
            if target == 0:
                label = 0
            else:
                label = 1
        else:
            assert False, 'Theres no such dataset...'
        
        return flow, label, target

    def __len__(self):
        return len(self.samples)
    
    
    def data_cnt_per_class(self):
        class_cnt = {label : 0 for label in self.classes}
        for i in range(len(self.targets)):
            class_cnt[str(self.targets[i])] += 1
        return class_cnt