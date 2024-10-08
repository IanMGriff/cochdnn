import h5py
import torch
import glob
import pickle
import numpy as np
# import psutil  # uncomment for tracking process in debug notebook 

class jsinV3_precombined_all_signals(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. As the authors for information on 
    # datafiles for training.
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int', 'signal/speaker_int', 'noise/labels_binary_via_int']

    def __init__(self, root, train=True, download=False, transform=None, batch_size=1, eval_max=3):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            if eval_max == -1:
                self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)
            else:
                self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:eval_max]

        self.datasets = [H5Dataset(h5_file, transform, self.target_keys, batch_size) for h5_file in self.all_hdf5_files]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load( open( "word_and_speaker_encodings_jsinv3.pckl", "rb" ))
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class jsinV3_precombined(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. 
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int']

    def __init__(self, root, train=True, download=False, transform=None, batch_size=1, eval_max=8):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:eval_max] # Just get one set of them

        self.datasets = [H5Dataset(h5_file, transform, self.target_keys, batch_size) for h5_file in self.all_hdf5_files]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map

class jsinV3_precombined_paired(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. 
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int']

    def __init__(self, root, train=True, download=False, transform=None, batch_size=1, eval_max=3):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            if eval_max == -1:
                self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)
            else:
                self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:eval_max]

        self.datasets = [H5DatasetPaired(h5_file, transform, self.target_keys, batch_size) for h5_file in self.all_hdf5_files]

        super().__init__(self.datasets)
        self.rotate_index = 0

    def _rotate_splits(self):
        for dataset in self.datasets:
            dataset._rotate_splits()
        self.rotate_index += 1 

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, batch_size):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.batch_size = batch_size

        # These TODOs are not implemented for the release. HDF5 files are 
        # already shuffled, so we can run through them directly. 
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal']) // self.batch_size # scale by batch size for dataloader

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys. 
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)# ["ndarray_data"]["signal"]
        # set up ix logic 
        start = index * self.batch_size
        end = start + self.batch_size

        # print(f"start ix: {start} on pid {psutil.Process().pid}") # uncomment for notebook print statements 
        # Before transforms, set the signal and the noise 
        signal = self.dataset['sources']['signal']['signal'][start:end]
        noise = self.dataset['sources']['noise']['signal'][start:end]

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signals = []
            for signal_, noise_ in zip(signal, noise):
                signal_, noise_ = self.transform(signal_, noise_)
                signals.append(signal_)
            signal = np.vstack(signals)
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split('/')
            target = self.dataset['sources'][target_paths[0]][target_paths[1]][start:end]
            if self.target_keys[0] == 'noise/labels_binary_via_int':
                target = target.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split('/')
                target[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][start:end]
                if target_key == 'noise/labels_binary_via_int':
                    target[target_key] = target[target_key].astype(np.float32)

        if self.transform is None:
            return signal, noise, target 
        
        return signal, target

    def __len__(self):
        return self.dataset_len


class H5DatasetPaired(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys, batch_size):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        self.batch_size = batch_size

        # These TODOs are not implemented for the release. HDF5 files are 
        # already shuffled, so we can run through them directly. 
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal'])  
        if self.dataset_len % 2 == 1:
            self.dataset_len -= 1

        self.rotate_index = 0
        all_indices = list(range(self.dataset_len))
        self.split_1 = all_indices[::2]
        self.split_2 = all_indices[1::2]
        # scale dataset len after setting split indices 
        self.dataset_len = self.dataset_len // self.batch_size  # scale by batch size for dataloader (accessed in len method)

    def _rotate_splits(self):
        self.split_2 = self.split_2[1:] + self.split_2[:1]
        self.rotate_index += 1

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys. 
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)# ["ndarray_data"]["signal"]
      
        # set up ix logic 
        start = index * self.batch_size
        end = start + self.batch_size


        # Before transforms, set the signal and the noise 
        # signal_1 = self.dataset['sources']['signal']['signal'][index]
        # noise_1 = self.dataset['sources']['noise']['signal'][index]

        # signal_2 = self.dataset['sources']['signal']['signal'][(index + 1) % self.dataset_len]
        # noise_2 = self.dataset['sources']['noise']['signal'][(index + 1) % self.dataset_len]

        signal_1 = self.dataset['sources']['signal']['signal'][self.split_1[start:end]]
        noise_1 = self.dataset['sources']['noise']['signal'][self.split_1[start:end]]

        try:
            signal_2 = self.dataset['sources']['signal']['signal'][self.split_2[start:end]]
            noise_2 = self.dataset['sources']['noise']['signal'][self.split_2[start:end]]
        except IndexError:
            print(self.split_2[start:end])
            print(start, end)
            signal_2 = signal_1
            noise_2 = noise_1

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signal_11, noise = self.transform(signal_1, noise_1)
            signal_12, noise = self.transform(signal_1, noise_2)
            signal_21, noise = self.transform(signal_2, noise_1)
            signal_22, noise = self.transform(signal_2, noise_2)
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split('/')
            target_1 = self.dataset['sources'][target_paths[0]][target_paths[1]][self.split_1[start:end]]
            try:
                target_2 = self.dataset['sources'][target_paths[0]][target_paths[1]][self.split_2[start:end]]
            except IndexError:
                target_2 = target_1
            if self.target_keys[0] == 'noise/labels_binary_via_int':
                target_1 = target_1.astype(np.float32)
                target_2 = target_2.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target_1, target_2 = {}, {}
            for target_key in self.target_keys:
                target_paths = target_key.split('/')
                target_1[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][self.split_1[start:end]]
                try:
                    target_2[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][self.split_2[start:end]]
                except IndexError:
                    target_2[target_key] = target_1[target_key]
                if target_key == 'noise/labels_binary_via_int':
                    target_1[target_key] = target_1[target_key].astype(np.float32)
                    target_2[target_key] = target_2[target_key].astype(np.float32)

        return signal_11, signal_12, signal_21, signal_22, target_1, target_2

    def __len__(self):
        return self.dataset_len // 2
