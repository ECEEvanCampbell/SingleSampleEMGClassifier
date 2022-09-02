
import os
import torch
import numpy as np
import scipy.signal
import random
from copy import deepcopy

class Dataset:
    def __init__(self, args):
        # store some member variables from our settings yml
        self.data_location = args.data_dir
        self.sampling_frequency = args.sampling_frequency
        self.preprocessing = args.preprocessing
        self.preprocessing_dictionary = {}
        self.modalities = args.modalities

        self.classes = args.classes

        # this is what will be used by the network
        self.emg_data = []
        self.imu_data = []
        self.labels = []
        # in case we ever want to keep track of what data was included
        self.processed_files = []

    def prepare_dataset(self):
        # actually prepare the dataset
        # always assume the 'EMG' and 'IMU' folders are within the base folder specified by self.data_location
        
        if "EMG" in self.modalities:
            files_to_process = os.listdir(self.data_location + "/EMG")
            

            # prepare the data somehow
            for f in files_to_process:
                class_identifier = f[5:-8]
                class_id = self.classes.index(class_identifier)
                # find the class_id of the file

                file_data = np.loadtxt(self.data_location + '/EMG/' + f, delimiter=',')
                
                if self.preprocessing == "envelope":
                    file_data = self.preprocess(file_data)
                if isinstance(self.emg_data,list):
                    self.emg_data = torch.tensor(file_data)
                    self.labels   = torch.tensor([class_id] * file_data.shape[0])
                else:
                    self.emg_data = torch.vstack((self.emg_data, torch.tensor(file_data)))
                    self.labels = torch.concat((self.labels, torch.tensor([class_id] * file_data.shape[0])))
                self.processed_files += [f]



        if "IMU" in self.modalities:
            # if you want to load both at the same time, you'd need to check that they're ALWAYS loaded in the same order (they should be atm.)
            # just in case you want to play w/ the imu
            for f in files_to_process:
                class_identifier = f[5:-8]
                class_id = self.classes.index(class_identifier)

                file_data = np.loadtxt(self.data_location + '/IMU/' + f, delimiter=',')

                if isinstance(self.imu_data, list):
                    self.imu_data = torch.tensor(file_data)
                    if "EMG" not in self.modalities:
                        self.labels = torch.tensor([class_id]*file_data.shape[0]*4)
                else:
                    self.imu_data = torch.vstack((self.imu_data, torch.tensor(file_data)))
                    if "EMG" not in self.modalitles:
                        self.labels = torch.concat((self.labels, torch.tensor([class_id] * file_data.shape[0]* 4)))
                self.processed_files += [f]
            ## -- insert pretty much the same processing as EMG if you want to try it out

    def preprocess(self, signals):
        # if filter has not been setup
        if not len(self.preprocessing_dictionary.keys()):
            # TODO: we should probably do some filtering before we envelope, but its fine for now
            if self.preprocessing == 'envelope':
                self.preprocessing_dictionary['b'], self.preprocessing_dictionary['a'] = scipy.signal.butter(4, 5 / (self.sampling_frequency/2), 'low')

        if self.preprocessing == 'envelope':
            # we should us filtfilt, but lfilter is more reaslistic for not doing a backwards pass over the signals
            signals = np.abs(signals)
            return scipy.signal.lfilter(self.preprocessing_dictionary['b'], self.preprocessing_dictionary['a'], signals, axis=0)

    def shuffle(self):
        indices = np.arange(self.labels.shape[0])
        np.random.shuffle(indices)
        self.labels = self.labels[indices]
        if "EMG" in self.modalities:
            self.emg_data = self.emg_data[indices,:]
        if "IMU" in self.modalities:
            self.imu_data = self.imu_data[indices[::4],:]


    def split(self, ratio, slice):
        return_dataset = deepcopy(self)
        num_samples = len(self)
        ratio = np.cumsum(ratio)
        if slice == 'train':
            if "EMG" in return_dataset.modalities:
                return_dataset.emg_data = return_dataset.emg_data[0:int(num_samples * ratio[0]),:]
            if "IMU" in return_dataset.modalities:
                return_dataset.imu_data = return_dataset.imu_data[0:int(num_samples * ratio[0]//4),:]
            return_dataset.labels   = return_dataset.labels[0:int(num_samples * ratio[0])]
        elif slice == 'val':
            if "EMG" in return_dataset.modalities:
                return_dataset.emg_data = return_dataset.emg_data[int(num_samples * ratio[0]): int(num_samples * ratio[1]),:]
            if "IMU" in return_dataset.modalities:
                return_dataset.imu_data = return_dataset.imu_data[int(num_samples * ratio[0]//4): int(num_samples * ratio[1]//4),:]
            return_dataset.labels   = return_dataset.labels[int(num_samples * ratio[0]): int(num_samples * ratio[1])]
        elif slice == 'test':
            if "EMG" in return_dataset.modalities:
                return_dataset.emg_data = return_dataset.emg_data[int(num_samples * ratio[1]): int(num_samples * ratio[2]),:]
            if "IMU" in return_dataset.modalities:
                return_dataset.imu_data = return_dataset.imu_data[int(num_samples * ratio[1]//4): int(num_samples * ratio[2]//4),:]
            return_dataset.labels   = return_dataset.labels[int(num_samples * ratio[1]): int(num_samples * ratio[2])]
        return return_dataset

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data = []
        if "EMG" in self.modalities:
            data += self.emg_data[idx,:].tolist()
        if "IMU" in self.modalities:
            data += self.imu_data[idx//4,:].tolist()
        data = torch.tensor(data)
        labels = self.labels[idx]
        return data, labels