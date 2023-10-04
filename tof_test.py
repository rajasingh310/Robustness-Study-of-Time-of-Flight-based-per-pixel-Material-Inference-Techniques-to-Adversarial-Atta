"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from util import tof_util
import torchvision
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()

    g_input = []
    true_labels = []
    real = []
    fake = []
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        g_input.append(torch.squeeze(model.real_C))
        true_labels.append(torch.squeeze(model.real_D))
        real.append(torch.squeeze(model.real_B))
        fake.append(torch.squeeze(model.fake_B))

    g_input = torch.stack(g_input, dim=0)
    true_labels = torch.stack(true_labels, dim=0)
    real = torch.stack(real, dim=0)
    fake = torch.stack(fake, dim=0)


    if opt.vis_tof_imgs_ON:
        from matplotlib.colors import ListedColormap, Normalize

        # Define your list of 15 colors
        colors = ['blue', 'yellow', 'red', 'green', 'purple', 'orange', 'cyan', 'pink', 'brown', 'gray', 'lime', 'teal',
                'magenta', 'navy', 'olive']

        # Create a colormap using the ListedColormap
        cmap = ListedColormap(colors)
        norm = Normalize(vmin=0, vmax=15)

        mat_classifier = tof_util.MaterialDetectionModel(num_materials=5).to('cuda')
        trained_model = torch.load("/home/ads/g050939/Downloads/mr_singh_thesis/pytorch-CycleGAN-and-pix2pix/util/mat_detect_2d_model.pth")
        mat_classifier.load_state_dict(trained_model)
        mat_classifier.eval()

        directory = 'tof_results'
        parent = '/home/ads/g050939/Downloads/mr_singh_thesis/pytorch-CycleGAN-and-pix2pix'

        path = os.path.join(parent, directory)

        if not os.path.exists(path):
            os.mkdir(path)
        
        for i in range(len(real)):

            # save the 2d image
            fig, ax = plt.subplots()
            ax.imshow(g_input[i, ...].to('cpu').numpy(), cmap='gray')
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(path + '/g_input_' + str(i) + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()

            image = true_labels[i, ...].to('cpu').numpy()
            rgb_data = np.squeeze(cmap(norm(image)))

            # save the 2d image
            fig, ax = plt.subplots()
            ax.imshow(rgb_data, cmap=cmap)
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(path + '/true_labels_' + str(i) + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()


            pred_real = tof_util.MaterialDetect(real[i, ...].to('cuda'), mat_classifier)
            image = pred_real.preds.unsqueeze(0).to('cpu').numpy()
            rgb_data = np.squeeze(cmap(norm(image)))

            # save the 2d image
            fig, ax = plt.subplots()
            ax.imshow(rgb_data, cmap=cmap)
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(path + '/real_labels_' + str(i) + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()

            pred_fake = tof_util.MaterialDetect(fake[i, ...].to('cuda'), mat_classifier)
            image = pred_fake.preds.unsqueeze(0).to('cpu').numpy()
            rgb_data = np.squeeze(cmap(norm(image)))

            # save the 2d image
            fig, ax = plt.subplots()
            ax.imshow(rgb_data, cmap=cmap)
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(path + '/fake_labels_' + str(i) + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()
                
    pass

