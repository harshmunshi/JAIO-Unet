import torch
import numpy as np
from os.path import join
import os, sys
from PIL import Image
import collections
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# In order to use the cross entropy loss, we need the target vector in the form of N dim vector

# define a palatte for converting the mask into one hot encoded map

# palette = {
#     'background':[0, 0, 0],
#     'aeroplane':[128, 0, 0],
#     'bicycle':[0, 128, 0],
#     'bird':[128, 128, 0],
#     'boat':[0, 0, 128],
#     'bottle':[128, 0, 128],
#     'bus':[0, 128, 128],
#     'car':[128, 128, 128],
#     'cat':[64, 0, 0],
#     'chair':[192, 0, 0],
#     'cow':[64, 128, 0],
#     'diningtable':[192, 128, 0],
#     'dog':[64, 0, 128],
#     'horse':[192, 0, 128],
#     'motorbike':[64, 128, 128],
#     'person':[192, 128, 128],
#     'pottedplant':[0, 64, 0],
#     'sheep':[128, 64, 0],
#     'sofa':[0, 192, 0],
#     'train':[128, 192, 0],
#     'tvmonitor':[0, 64, 128],
#     'void':[224, 224, 192]}

palette = [[0, 0, 0],
           [128, 0, 0],
           [0, 128, 0],
           [128, 128, 0],
           [0, 0, 128],
           [128, 0, 128],
           [0, 128, 128],
           [128, 128, 128],
           [64, 0, 0],
           [192, 0, 0],
           [64, 128, 0],
           [192, 128, 0],
           [64, 0, 128],
           [192, 0, 128],
           [64, 128, 128],
           [192, 128, 128],
           [0, 64, 0],
           [128, 64, 0],
           [0, 192, 0],
           [128, 192, 0],
           [0, 64, 128],
           [224, 224, 192]]


class VOCLoader(Dataset):
    """
    The ground truth is given in the form of RGB images.
    So we approach the target creation as following steps:
    1. Normalize the dataset (for faster convergence, [see batch normalization]).
    2. Normalize the image.
    3. Convert the mask from RGB to label_mask format.
    4. In __getitem__ return normalized image as image and label_mask for that image as the target
    """
    def __init__(self, root_dir, test_mode=False, split="train", augmentations=None):
        super(VOCLoader, self).__init__()
        # read the images as well as the labels, convert the labels in one hot format
        # This is the root dir <VOC_ROOT/VOC2012>
        self.root_dir = root_dir
        self.test_mode = test_mode
        self.n_class = 21
        self.split = split
        self.augmentations = augmentations
        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict()
        # join the paths <ImageSets/Segmentation> to the root directory
        if not self.test_mode:
            for split in ["train", "val", "trainval"]:
                path = join(self.root_dir, "ImageSets/Segmentation", split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.strip() for id_ in file_list]
                self.files[split] = file_list
        
        # normalization for image (in order to get better gradient flow towards the minima)
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ]
        )
    

    def encode_mask(self, mask):
        """
        Encode the label masks from RGB to class one hot encoding
        """
        mask = np.array(mask)
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for idx, label in enumerate(palette):
            # get the indices and start filling the blank image based on the RGB color from the palette
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = idx
        label_mask = label_mask.astype(int)
        return label_mask


    def preprocess(self, name):
        # name is the path to that image and the label
        im_name = name + ".jpg"
        seg_name = name + ".png"
        label_path = join(self.root_dir, "SegmentationClass", seg_name)
        img_path = join(self.root_dir, "JPEGImages", im_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")

        # transform the image and encode the mask
        # NOTE: Ideally at this point we have to resize the input image but
        # since the input and output dimensions are same for the respective masks, it doesn't matter
        img = self.tf(img)
        mask = self.encode_mask(mask)

        # convert the mask to a torch tensor
        mask = torch.from_numpy(mask).long()
        mask[mask == 255] = 0
        return img, mask
    
    
    def __getitem__(self, idx):
        """
        fetch the image and annotation by index
        """
        if not self.test_mode: 
            # If it's training
            #index = np.random.choice(len(self.files["train"]), 1)[0]
            name = self.files["train"][idx]
        elif self.test_mode==True:
            #index = np.random.choice(len(self.files["val"]), 1)[0]
            name = self.files["val"][idx]
        
        image, mask = self.preprocess(name)
        return image, mask    

    def __len__(self):
        return len(self.files[self.split])    

# # load a sample image
# fixed_path = join()
# sample_img_path = fixed_path + "2010_005830.png"
# img = Image.open(sample_img_path).convert('RGB')
# to_mask(img)

def unit_test():
    root_dir = "/home/harshmunshi/JAIO-Unet/Unet/data/VOCdevkit/VOC2012"
    loader = VOCLoader(root_dir)
    trainloader = DataLoader(loader, batch_size=1)
    for i, data in enumerate(trainloader):
        print(i, data)


if __name__=="__main__":
    unit_test()
    print("============Unit Test Passed==============")