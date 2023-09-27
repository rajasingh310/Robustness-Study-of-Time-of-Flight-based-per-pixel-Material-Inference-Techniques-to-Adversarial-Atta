"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
import tifffile as tiff


class ToFpix2pixDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)

        '''
        
        Directory A in the opt.dataroot directory holds the images for the generator
        Directory B in the opt.dataroot directory holds the images for the Discriminator
        Directory C in the opt.dataroot directory holds the images same as the images in directory A but in its non normalized form.
            It is basically used for visualization purpose. It is neither used for training nor testing.
        Directory D in the opt.dataroot directory holds the labels. These labels can help in choosing the best performing model among the saved (trained) models.
        
        '''

        self.A_path = os.path.join(opt.dataroot, 'A', opt.phase)  # for generator input
        self.B_path = os.path.join(opt.dataroot, 'B', opt.phase)    # for discriminator input
        self.C_path = os.path.join(opt.dataroot, 'C', opt.phase)  # for visualization of the generator input image
        self.D_path = os.path.join(opt.dataroot, 'D', opt.phase)    # for verifying the quality of the model

        self.A_paths = sorted(make_dataset(self.A_path, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.B_path, opt.max_dataset_size))
        self.C_paths = sorted(make_dataset(self.C_path, opt.max_dataset_size))
        self.D_paths = sorted(make_dataset(self.D_path, opt.max_dataset_size))

        if opt.direction == 'BtoA':
            self.A_paths, self.B_paths = self.B_paths, self.A_paths
            opt.input_nc, opt.output_nc = opt.output_nc, opt.input_nc

        opt.G_input_shape = np.array(tiff.imread(self.A_paths[0])).shape


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        data_A = np.array(tiff.imread(self.A_paths[index]))    # needs to be a tensor
        data_B = np.array(tiff.imread(self.B_paths[index]))    # needs to be a tensor
        data_C = np.array(tiff.imread(self.C_paths[index]))    # needs to be a tensor
        data_D = np.array(tiff.imread(self.D_paths[index]))    # needs to be a tensor
        
        data_A = data_A[np.newaxis, ...]
        data_C = data_C[np.newaxis, ...]
        data_D = data_D[np.newaxis, ...]

        return {'A': data_A, 'B': data_B, 'C': data_C, 'D': data_D, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.A_paths)
