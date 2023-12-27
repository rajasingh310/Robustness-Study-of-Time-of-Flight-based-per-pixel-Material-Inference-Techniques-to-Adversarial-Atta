import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


def slide_window(tof_image):

    pad_image = F.pad(tof_image.unsqueeze(0), (2, 2, 2, 2), "reflect")

    window_row = 5
    window_col = 5
    output_size = (pad_image.shape[-3], window_row, window_col)
    stride = (pad_image.shape[-2] * pad_image.shape[-1], pad_image.shape[-1], 1)

    images = []
    j = 0
    i = 0
    for _ in range(tof_image.shape[-2] * tof_image.shape[-1]):

        if j < tof_image.shape[-1]:
            images.append(torch.as_strided(pad_image, output_size, stride, i))
            j += 1
            i += 1

        else:
            j = 0
            i += window_col
            images.append(torch.as_strided(pad_image, output_size, stride, i))

    return torch.stack(images, dim=0)

def image_filtering(window_images, std_dev1, std_dev2):

    std_dev2 = std_dev2.unsqueeze(1).unsqueeze(2).to('cuda')

    row_indices = torch.arange(window_images.shape[-2])
    col_indices = torch.arange(window_images.shape[-1])

    r, c = torch.meshgrid(row_indices, col_indices)

    centreR = len(row_indices) // 2
    centreC = len(col_indices) // 2

    r = r - centreR
    c = c - centreC

    dist_mat = torch.exp(-(torch.sqrt(r ** 2 + c ** 2)) / (2 * (std_dev1 ** 2))).to('cuda')

    filtered_images = []

    for i in range(len(window_images)):
        win_img = window_images[i, ...]

        centre_row_index = win_img.shape[-2] // 2
        centre_col_index = win_img.shape[-1] // 2

        win_centre_pixels = win_img[:, centre_row_index, centre_col_index]. unsqueeze(1).unsqueeze(2)
        range_mat = (win_img - win_centre_pixels) ** 2
        range_mat = torch.exp(-(range_mat / (2 * (std_dev2 ** 2))))

        filtered_images.append(dist_mat * range_mat)
    
    return torch.stack(filtered_images, dim=0)
    
def k_nearest_neighbor(win_images, filtered_images, k):
    num_images, channels, height, width = win_images.shape

    centreR = height // 2
    centreC = width // 2

    r, c = torch.meshgrid(torch.arange(height), torch.arange(width))
    r = torch.ravel(r).to('cuda')
    c = torch.ravel(c).to('cuda')

    knn_images = []

    for i in range(num_images):
        b_image = filtered_images[i, ...] - (filtered_images[1, :, centreR, centreC]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        b_image = torch.ravel(torch.norm(torch.squeeze(b_image), dim=0, p=2))

        indices = torch.argsort(b_image)
        img = win_images[i, :, r[indices[:k]], c[indices[:k]]]

        knn_images.append(img)

    return torch.stack(knn_images, dim=0).unsqueeze(2)


class ToFImagePreProcessing():

    def __init__(self, image):    # shape of the image (m, 16, 172, 224)

        self.window_images = slide_window(image)

        dataset_dir = "dataset_3"
        path_pre_trained_classifier_and_std_dev = "/home/ads/g050939/Downloads/mr_singh_thesis/pytorch-CycleGAN-and-pix2pix/util/tof_pre_trained_model_and_std_dev/"

        std_dev1 = 7.5
        std_dev2 = torch.load(path_pre_trained_classifier_and_std_dev + dataset_dir + "/std_dev.pth")
        self.bilateral_images = image_filtering(self.window_images, std_dev1, std_dev2)    # bilateral filtering

        self.knn = k_nearest_neighbor(self.window_images, self.bilateral_images , 9)   

class MaterialDetectionModel(nn.Module):
    def __init__(self, num_materials=5):
        super(MaterialDetectionModel, self).__init__()

        self.input_shape = (16, 1, 9)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.input_shape[0]),
            nn.Flatten(),
            nn.Linear(self.input_shape[0]*self.input_shape[1]*self.input_shape[2], 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(inplace=True),
            nn.Linear(250, num_materials),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class MaterialDetect():
    def __init__(self, images, model):

        images = ToFImagePreProcessing(images)

        with torch.no_grad():
            self.preds = torch.argmax(model(images.knn), 1).reshape(172, 224)