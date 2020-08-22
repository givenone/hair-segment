
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LfwDataset(Dataset):
    def __init__(self, root_dir, train=True, joint_transforms=None,
                 image_transforms=None, mask_transforms=None, gray_image=False):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): True if to add gray images
        """

        txt_file = 'parts_train_val.txt' if train else 'parts_test.txt'
        txt_dir = os.path.join(root_dir, txt_file)
        name_list = LfwDataset.parse_name_list(txt_dir)
        img_dir = os.path.join(root_dir, 'lfw_funneled')
        mask_dir = os.path.join(root_dir, 'parts_lfw_funneled_gt_images')

        self.img_path_list = [os.path.join(img_dir, elem[0], elem[1]+'.jpg') for elem in name_list]
        self.mask_path_list = [os.path.join(mask_dir, elem[1]+'.ppm') for elem in name_list]
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.gray_image = gray_image

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)

        mask_arr = np.array(mask)
        #hair_face_map = np.zeros(mask_arr.shape[0:2])
        hair_face_map = np.full(mask_arr.shape[0:2], [1, 0, 0]) # background = [1,0,0]

        face_map = mask_arr == np.array([0, 255, 0])
        face_map = np.all(face_map, axis=2).astype(np.float32)
        
        hair_map = mask_arr == np.array([255, 0, 0])
        hair_map = np.all(hair_map, axis=2).astype(np.float32)
        
        hair_face_map[np.where(hair_map == 1)] = [0, 1, 0] # hair = [0, 1, 0]
        hair_face_map[np.where(face_map == 1)] = [0, 0, 1] # face = [0, 0, 1]
        
        mask = Image.fromarray(hair_face_map)
        mask = mask.convert('L')

        #mask.save("./overlay/temp" + str(idx) + ".png")
        #print("이미지 생성")
        # mask = rgb_to_ternary(mask)

        # mask = LfwDataset.rgb2binary(mask)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)

        if self.gray_image:
            gray = img.convert('L')
            gray = np.array(gray,dtype=np.float32)[np.newaxis,]/255
            return img, mask, gray

        return img, mask

    def __len__(self):
        return len(self.mask_path_list)
    
    @staticmethod
    def rgb2binary(mask):
        """
        transforms RGB mask image to binary hair mask image.
        """
        
        mask_arr = np.array(mask)
        mask_map = mask_arr == np.array([0, 255, 0])
        mask_map = np.all(mask_map, axis=2).astype(np.float32)
        return Image.fromarray(mask_map)
    
    @staticmethod
    def rgb_to_ternary(mask):
        mask_arr = np.array(mask)
        mask_arr = np.array(mask)
        hair_face_map = np.zeros(mask_arr.shape[0:2])
        
        face_map = mask_arr == np.array([0, 255, 0])
        face_map = np.all(face_map, axis=2).astype(np.float32)
        
        hair_map = mask_arr == np.array([255, 0, 0])
        hair_map = np.all(hair_map, axis=2).astype(np.float32)
        
        hair_face_map[np.where(hair_map == 1)] = 128
        hair_face_map[np.where(face_map == 1)] = 255
        
        return Image.fromarray(hair_face_map)

    @staticmethod
    def parse_name_list(fp):
        with open(fp, 'r') as fin:
            lines = fin.readlines()
        parsed = list()
        for line in lines:
            name, num = line.strip().split(' ')
            num = format(num, '0>4')
            filename = '{}_{}'.format(name, num)
            parsed.append((name, filename))
        return parsed

