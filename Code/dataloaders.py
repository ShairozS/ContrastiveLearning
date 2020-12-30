from torch.utils.data import Dataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


def augment(imgtensor):
    '''
    Apply a content preserving augmentation to an image as a tensor 
    '''
    from torchvision.transforms.functional import hflip
    
    aug_img = hflip(imgtensor)
    return(aug_img)


def np2tensor(nparray, resize_to=None):
    
    '''
    Convert a (CHW) numpy array to (HWC) torch Tensor
    '''
    if resize_to is not None:
        nparray = cv2.resize(nparray, dsize=resize_to, interpolation=cv2.INTER_CUBIC)
                             
    out_array = np.moveaxis(nparray, 2, 0)
    out_array = out_array / 255

    return(torch.Tensor(out_array))


class LFWContrastiveDataset(Dataset):



      def __init__(self, data_path = r'./Data/Train/', resize_to=None):
      
            '''
            Initialize and sort all datapaths here so they
            can be retrieved easily for loading by the __getitem__() method
            '''
            
            self.data_path = data_path
            self.img_paths = [os.path.join(data_path, p) for p in os.listdir(data_path)]
            self.resize_to = resize_to
      

      def __len__(self):
            '''
            Calculate the total amount of examples and return it here
            '''
            return(len(self.img_paths))
      

      def __getitem__(self, idx):
            
            '''
            Load and return the idx-th data example and label 
            '''
            # Initialize the out dictionary
            out_dict = {"x1": [], "x2":[], "label":[]}
            
            # Sample the anchor image from the idx-th class
            anchor_class = self.img_paths[idx]
           
            
            # How many images are there for the idx-th class
            anchor_class_imgs = [os.path.join(anchor_class,p) for p in os.listdir(anchor_class)]
            
            # Grab the anchor (1st) image in the class folder
            anchor_img = plt.imread(anchor_class_imgs[0])
            
            
            ################ Anchor-positive sampling ###################
            # If there are more (similar images in the folder)
            if len(anchor_class_imgs) > 1:
                
                #Gather all the anchor-positive pairs
                for pos in range(1, len(anchor_class_imgs)):
                    
                    pos_img = plt.imread(anchor_class_imgs[pos])
                    
                    out_dict["x1"].append(np2tensor(anchor_img, resize_to = self.resize_to))
                    out_dict["x2"].append(np2tensor(pos_img, resize_to = self.resize_to))
                    out_dict["label"].append(torch.Tensor([1.]))
            
            # If there aren't any more similar images in the folder
            else:
                
                # Create an augmented anchor-positive pair
                out_dict["x1"].append(np2tensor(anchor_img, resize_to = self.resize_to))
                out_dict["x2"].append(augment(np2tensor(anchor_img, resize_to = self.resize_to)))
                out_dict["label"].append(torch.Tensor([1.]))
            
            ############### Anchor-negative sampling ##################
            
                
            # Sample a dissimilar class
            neg_class = self.img_paths[np.random.choice(len(self.img_paths))]
            neg_class_imgs = [os.path.join(neg_class, p) for p in os.listdir(neg_class)]
            
            # Create anchor-negative examples from that class
            for neg in range(len(neg_class_imgs)):
                
                neg_img = plt.imread(neg_class_imgs[neg])
                
                out_dict["x1"].append(np2tensor(anchor_img, resize_to = self.resize_to))
                out_dict["x2"].append(np2tensor(neg_img, resize_to = self.resize_to))
                out_dict["label"].append(torch.Tensor([0.]))
            
            # Batch the tensors in the out_dict
            for key in out_dict.keys():
                out_dict[key] = torch.stack(out_dict[key], axis=0)
            
            
            return(out_dict)
