from torch.utils import data

import numpy as np
import torch
import h5py
import os

project_root = os.path.dirname(os.path.dirname(__file__))
data_folder = os.path.join(project_root, 'dataset')

class PhC2D(data.Dataset):
    def __init__(self, train: bool) -> None:
        if train == True:
            indlist = np.load(os.path.join(data_folder, 'train_indices.npy'))
        else:
            indlist = np.load(os.path.join(data_folder, 'test_indices.npy'))
        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []

        with h5py.File(os.path.join(data_folder, 'bandgap_5classes_4.h5'), 'r') as f:
            for memb in indlist:
                input = f['input_uc/'+str(memb)][()]
                y = f['class/'+str(memb)][()]
                self.x_data.append(input)
                self.y_data.append(y)

        # normalize x data
        x_data_mean = 9.3349
        x_data_std = 6.0522

        self.x_data = (np.array(self.x_data).astype('float32') - x_data_mean) / x_data_std # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('long')

        self.x_data = torch.tensor(self.x_data)
        self.y_data = torch.tensor(self.y_data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple:
        return self.x_data[index], self.y_data[index]
    
class PhC2DBandgap(data.Dataset):
    # The class boundaries are:
    # Class 0: BG = 0%
    # Class 1: 0% < BG < 5%
    # Class 2: 5% < BG < 12%
    # Class 3: 12% < BG < 20%
    # Class 4: 20% < BG 
    def __init__(self, train: bool) -> None:
        if train == True:
            indlist = np.load(os.path.join(data_folder, 'train_indices.npy'))
        else:
            indlist = np.load(os.path.join(data_folder, 'test_indices.npy'))
        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []

        with h5py.File(os.path.join(data_folder, 'mf1-tm-32k.h5'), 'r') as bandgap_value_file:
            with h5py.File(os.path.join(data_folder, 'bandgap_5classes_4.h5')) as bandgap_class_file:
                for memb in indlist:
                    input = bandgap_class_file['input_uc/'+str(memb)][()]
                    regression_index = bandgap_class_file['oldindex/'+str(memb)][()]
                    bandgap_info = bandgap_value_file['mpbcal/bandgap/'+str(regression_index)][()]
                    if len(bandgap_info) == 0:
                        bandgap = 0.0
                    else:
                        bandgaps = bandgap_info[:,0]
                        bandgap = np.max(bandgaps)
                    self.x_data.append(input)
                    self.y_data.append(bandgap)

        # normalize x data
        x_data_mean = 9.3349
        x_data_std = 6.0522

        self.x_data = (np.array(self.x_data).astype('float32') - x_data_mean) / x_data_std # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('float32')

        self.x_data = torch.tensor(self.x_data)
        self.y_data = torch.tensor(self.y_data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple:
        return self.x_data[index], self.y_data[index]


class PhC2DBandgapQuickLoad(data.Dataset):
    def __init__(self, train: bool) -> None:
        data_file = h5py.File(os.path.join(data_folder, 'bandgap_values.h5'), 'r')
        
        if train:
            self.x_data = torch.tensor(np.array(data_file.get('train_inputs')))
            self.y_data = torch.tensor(np.array(data_file.get('train_targets')))
        else:
            self.x_data = torch.tensor(np.array(data_file.get('test_inputs')))
            self.y_data = torch.tensor(np.array(data_file.get('test_targets')))
        
        self.len = len(self.y_data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple:
        return self.x_data[index], self.y_data[index]