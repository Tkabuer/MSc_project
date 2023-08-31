from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import ImageCms
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage
import glob

def ResizeCrop(image, sz, div_factor):

    image_size = image.size
    image = transforms.Resize([image_size[1] // div_factor, \
                                   image_size[0] // div_factor])(image)
    
    if image.size[1] < sz[0] or image.size[0] < sz[1]:
        # image size smaller than crop size, zero pad to have same size
        image = transforms.CenterCrop(sz)(image)
    else:
        image = transforms.RandomCrop(sz)(image)
    
    return image

def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image

def MS_transform(image):
#   MS Transform
    image = np.array(image).astype(np.float32)
    window = gen_gauss_window(3, 7/6)
    image[:,:,0] = compute_MS_transform(image[:,:,0], window)
    image[:,:,0] = (image[:,:,0] - np.min(image[:,:,0]))/(np.ptp(image[:,:,0])+1e-3)
    image[:,:,1] = compute_MS_transform(image[:,:,1], window)
    image[:,:,1] = (image[:,:,1] - np.min(image[:,:,1]))/(np.ptp(image[:,:,1])+1e-3)
    image[:,:,2] = compute_MS_transform(image[:,:,2], window)
    image[:,:,2] = (image[:,:,2] - np.min(image[:,:,2]))/(np.ptp(image[:,:,2])+1e-3)
    
    image = Image.fromarray((image*255).astype(np.uint8))
    return image

def colorspaces(im, val):
    if val == 0:
        im = transforms.RandomGrayscale(p=1.0)(im)
    elif val == 1:
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p  = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        im = ImageCms.applyTransform(im, rgb2lab)
    elif val == 2:
         im = im.convert('HSV')
    elif val == 3:
         im = MS_transform(im)
    return im

class image_data(Dataset):
    def __init__(self, file_path, image_size = (128, 128), transform = True): #image_size = (256, 256)
        self.fls = pd.read_csv(file_path)
        self.image_size = image_size
        
        self.tranform_toT = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                ])
    
    def __len__(self):
        return len(self.fls)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path = self.fls.iloc[idx]['name']

        # cat 12 frames
        images = torch.empty([0, 128, 128]) # 256, 256
        images_2 = torch.empty([0, 128, 128])

        files = glob.glob(path)
        for file in files:
            image = Image.open(file)
            colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
            image2 = colorspaces(image, colorspace_choice)
            colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
            image = colorspaces(image, colorspace_choice)
            # scaling transform and random crop
            div_factor = np.random.choice([1,2],1)[0]
            image2 = ResizeCrop(image2, self.image_size, div_factor)
            image = ResizeCrop(image, self.image_size, 3 - div_factor)

            image = self.tranform_toT(image)
            image2 = self.tranform_toT(image2)
            images = torch.cat((images, image), dim=0)
            images_2 = torch.cat((images_2, image2), dim=0)

        def get_label(target, label_list):
            return np.array([1 if target in s else 0 for s in label_list]).astype(np.float32)

        video_index = path.split('/')[-1][-2]
        dis_name = path.split('/')[3].split('_')[1-eval(video_index)]
        labels = ['adaconf', 'bmbc', 'cain', 'ctx', 'cycle', 'dain', 'pyrwarp', 'rrin', 'sepconv_l1', 'super_slomo']

        label = get_label(dis_name, labels)

        # img_name = self.fls.iloc[idx]['File_names'].rstrip()
        # image_orig = Image.open(img_name)
        
        # if image_orig.mode == 'L':
        #     image_orig = np.array(image_orig)
        #     image_orig = np.repeat(image_orig[:,:,None],3,axis=2)
        #     image_orig = Image.fromarray(image_orig)
        # elif image_orig.mode != 'RGB':
        #     image_orig = image_orig.convert('RGB')
            
        
        # # Data augmentations
        
        # # scaling transform and random crop
        # div_factor = np.random.choice([1,2],1)[0]
        # image_2 = ResizeCrop(image_orig, self.image_size, div_factor)
        
        # # change colorspace
        # colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
        # image_2 = colorspaces(image_2, colorspace_choice)
        # image_2 = self.tranform_toT(image_2)
        
        # # scaling transform and random crop
        # image = ResizeCrop(image_orig, self.image_size, 3 - div_factor)
        
        # # change colorspace
        # colorspace_choice = np.random.choice([0,1,2,3,4],1)[0]
        # image = colorspaces(image, colorspace_choice)
        # image = self.tranform_toT(image)
        
        # # read distortion class, for authentically distorted images it will be 0
        # label = self.fls.iloc[idx]['labels']
        # label = label[1:-1].split(' ')
        # label = np.array([t.replace(',','') for t in label]).astype(np.float32)
        
        return images, images_2, label
