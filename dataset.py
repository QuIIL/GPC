import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random
import json

class ImageLabel(Dataset):
    def __len__(self) -> int:
        return len(self.pair_list)
    
    def train_augmentors(self):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return input_augs

    def __getitem__(self, index):
        img_path, label = self.pair_list[index]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (512,512))

        train_augmentors = self.train_augmentors()
        img_tensor = train_augmentors.augment_image(image)
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1) # C,H,W

        return img_path, img_tensor, label

    def __init__(self, pair_list):
        self.pair_list = pair_list

class ImageCaptionDataset(Dataset):
    def __len__(self) -> int:
        return len(self.pair_list)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def train_augmentors(self):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return input_augs

    def __getitem__(self, index):
        img_path, caption = self.pair_list[index]
        tokens, mask = self.pad_tokens(self.tokens_list[index])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.resize,self.resize))

        train_augmentors = self.train_augmentors()
        aug_image = train_augmentors.augment_image(image)
        img_tensor = torch.tensor(aug_image, dtype=torch.float32).permute(2,0,1) # C,H,W

        return img_path, tokens, mask, img_tensor, caption

    def __init__(self, 
                 pair_list,  
                 prefix_length,
                 tokenizer,
                 resize=512):
        self.prefix_length = prefix_length
        self.tokenizer = tokenizer
        self.pair_list = pair_list
        self.resize = resize

        self.tokens_list = []

        all_len = []
        for data in self.pair_list:
            tokens = self.tokenizer(data[1], return_tensors='pt', padding=True).input_ids.squeeze(0)
            all_len.append(len(tokens))
            self.tokens_list.append(tokens)
        mean_len = sum(all_len) / len(all_len)
        self.max_seq_len = min(int(mean_len + np.std(all_len) * 10), int(np.max(all_len)))

    def __len__(self) -> int:
        return len(self.pair_list)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def train_augmentors(self):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return input_augs

    def __getitem__(self, index):
        img_path, caption = self.pair_list[index]
        tokens, mask = self.pad_tokens(self.tokens_list[index])

        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.resize,self.resize))

        train_augmentors = self.train_augmentors()
        img_tensor = train_augmentors.augment_image(image)
        img_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1) # C,H,W

        return img_path, tokens, mask, img_tensor, caption

    def __init__(self, 
                 pair_list,  
                 prefix_length,
                 tokenizer,
                 resize=512):
        self.prefix_length = prefix_length
        self.tokenizer = tokenizer
        self.pair_list = pair_list
        self.resize = resize

        self.tokens_list = []

        all_len = []
        for data in self.pair_list:
            tokens = self.tokenizer(data[1], return_tensors='pt').input_ids.squeeze(0)
            all_len.append(len(tokens))
            self.tokens_list.append(tokens)
        mean_len = sum(all_len) / len(all_len)
        self.max_seq_len = min(int(mean_len + np.std(all_len) * 10), int(np.max(all_len)))


    def __init__(self, pair_list, tokenizer, args):
        self.scales = args.image_scale
        if args.feature_concat_order == 'small_to_big':
            self.scales.sort(reverse=True)
        elif args.feature_concat_order == 'big_to_small':
            self.scales.sort(reverse=False)
        self.prefix_length = args.prefix_length * (len(args.image_scale)+1)
        super().__init__(pair_list, self.prefix_length, tokenizer)

    def __getitem__(self, index):
        img_path, caption = self.pair_list[index]
        tokens, mask = self.pad_tokens(self.tokens_list[index])

        img_scale_tensors = []

        # original scale forward
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.resize,self.resize))
        train_augmentors = self.train_augmentors()
        aug_image = train_augmentors.augment_image(image)
        img_tensor = torch.tensor(aug_image.copy(), dtype=torch.float32).permute(2,0,1) # C,H,W
        img_scale_tensors.append(img_tensor)

        # new scales forward
        for scale in self.scales:
            new_d = int(self.resize * scale)
            scaled_image = cv2.resize(image, (new_d, new_d))
            aug_scaled_image = train_augmentors.augment_image(scaled_image)
            img_tensor = torch.tensor(aug_scaled_image.copy(), dtype=torch.float32).permute(2,0,1) # C,H,W
            img_scale_tensors.append(img_tensor)

        return img_path, tokens, mask, img_scale_tensors, caption

def prepare_colon(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '0': 'benign.',
            '1': 'well differentiated cancer.',
            '2': 'moderately differentiated cancer.',
            '3': 'poorly differentiated cancer.',
        }
        label = path.split('_')[-1].split('.')[0]
        if label_type == 'caption':
            return mapping_dict[label]
        else:
            return int(path.split('_')[-1].split('.')[0])
    
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [map_label_caption(file_path) for file_path in file_list]

        return list(zip(file_list, label_list))

    data_root_dir = '/home/compu/anhnguyen/dataset/KBSMC_512'
    set_tma01 = load_data_info('%s/tma_01/*.jpg' % data_root_dir)
    set_tma02 = load_data_info('%s/tma_02/*.jpg' % data_root_dir)
    set_tma03 = load_data_info('%s/tma_03/*.jpg' % data_root_dir)
    set_tma04 = load_data_info('%s/tma_04/*.jpg' % data_root_dir)
    set_tma05 = load_data_info('%s/tma_05/*.jpg' % data_root_dir)
    set_tma06 = load_data_info('%s/tma_06/*.jpg' % data_root_dir)
    set_wsi01 = load_data_info('%s/wsi_01/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi02 = load_data_info('%s/wsi_02/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi03 = load_data_info('%s/wsi_03/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_tma01 + set_tma02 + set_tma03 + set_tma05 + set_wsi01
    valid_set = set_tma06 + set_wsi03
    test_set = set_tma04 + set_wsi02

    return train_set, valid_set, test_set

def prepare_colon_test_2(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '1': 'benign.',
            '2': 'well differentiated cancer.',
            '3': 'moderately differentiated cancer.',
            '4': 'poorly differentiated cancer.',
        }
        label = path.split('_')[-1].split('.')[0]

        if label_type == 'caption':
            return mapping_dict[label]
        else:
            return int(label)-1

    def load_data_info_from_list(data_dir, path_list):
        file_list = []
        for WSI_name in path_list:
            pathname = glob.glob(f'{data_dir}/{WSI_name}/*/*.png')
            file_list.extend(pathname)
            label_list = [map_label_caption(file_path) for file_path in file_list]
        list_out = list(zip(file_list, label_list))

        return list_out

    data_root_dir = '/home/compu/anhnguyen/dataset/KBSMC_512_test2/KBSMC_test_2'
    wsi_list = ['wsi_001', 'wsi_002', 'wsi_003', 'wsi_004', 'wsi_005', 'wsi_006', 'wsi_007', 'wsi_008', 'wsi_009',
                'wsi_010', 'wsi_011', 'wsi_012', 'wsi_013', 'wsi_014', 'wsi_015', 'wsi_016', 'wsi_017', 'wsi_018',
                'wsi_019', 'wsi_020', 'wsi_021', 'wsi_022', 'wsi_023', 'wsi_024', 'wsi_025', 'wsi_026', 'wsi_027',
                'wsi_028', 'wsi_029', 'wsi_030', 'wsi_031', 'wsi_032', 'wsi_033', 'wsi_034', 'wsi_035', 'wsi_090',
                'wsi_092', 'wsi_093', 'wsi_094', 'wsi_095', 'wsi_096', 'wsi_097', 'wsi_098', 'wsi_099', 'wsi_100']

    test_set = load_data_info_from_list(data_root_dir, wsi_list)

    return test_set

def prepare_prostate_uhu_data(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '0': 'benign.',
            '1': 'grade 3 cancer.',
            '2': 'grade 4 cancer.',
            '3': 'grade 5 cancer.',
        }
        mapping_dict_2 = {
            0:0,
            1:4,
            2:5,
            3:6
        }
        label = path.split('_')[-1].split('.')[0]
        if label_type == 'caption':
            return mapping_dict[label]
        elif label_type == 'combine_dataset':
            temp = int(path.split('_')[-1].split('.')[0])
            return mapping_dict_2[temp]
        else:
            return int(label)

    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [map_label_caption(file_path) for file_path in file_list]
        return list(zip(file_list, label_list))

    data_root_dir = '/home/compu/doanhbc/datasets/prostate_harvard'
    data_root_dir_train = f'{data_root_dir}/patches_train_750_v0'
    data_root_dir_valid = f'{data_root_dir}/patches_validation_750_v0'
    data_root_dir_test = f'{data_root_dir}/patches_test_750_v0'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_valid)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    return train_set, valid_set, test_set

def prepare_prostate_ubc_data(label_type='caption'):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_dict = {
            0: 'benign.', 
            2: 'grade 3 cancer.', 
            3: 'grade 4 cancer.', 
            4: 'grade 5 cancer.'
        }
        mapping_dict_2 = {
            0:0,
            2:4,
            3:5,
            4:6
        }
        if label_type == 'caption':
            label_list = [label_dict[k] for k in label_list]
        elif label_type == 'combine_dataset':
            for i in range(len(label_list)):
                label_list[i] = mapping_dict_2[label_list[i]]
        else:
            for i in range(len(label_list)):
                if label_list[i] != 0:
                    label_list[i] = label_list[i] - 1

        return list(zip(file_list, label_list))
    
    data_root_dir = '/home/compu/doanhbc/datasets'
    data_root_dir_train_ubc = f'{data_root_dir}/prostate_miccai_2019_patches_690_80_step05_test/'
    test_set_ubc = load_data_info('%s/*/*.jpg' % data_root_dir_train_ubc)
    return test_set_ubc

def prepare_gastric(nr_classes=4, label_type='caption'):
    def load_data_info_from_list(path_list, gt_list, data_root_dir, label_type='caption'):
        mapping_dict = {
            0: 'benign.',
            1: 'tubular well differentiated cancer.',
            2: 'tubular moderately differentiated cancer.',
            3: 'tubular poorly differentiated cancer.',
            4: 'other'
        }

        mapping_dict_2 = {
            0:0,
            1:7,
            2:8,
            3:9,
            4:2
        }

        file_list = []
        for tma_name in path_list:
            pathname = glob.glob(f'{data_root_dir}/{tma_name}/*.jpg')
            file_list.extend(pathname)
        
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        if label_type == 'caption':
            label_list = [mapping_dict[gt_list[i]] for i in label_list]
        elif label_type == 'combine_dataset':
            label_list = [mapping_dict_2[gt_list[i]] for i in label_list]
        else:
            label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))
        if label_type == 'caption':
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] != 'other']
        elif label_type == 'combine_dataset':
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] != 2]
        else:
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < 4]

        return list_out

    def load_a_dataset(csv_path, gt_list, data_root_dir, data_root_dir_2, down_sample=True, label_type='caption'):
        df = pd.read_csv(csv_path).iloc[:, :3]
        train_list = list(df.query('Task == "train"')['WSI'])
        valid_list = list(df.query('Task == "val"')['WSI'])
        test_list = list(df.query('Task == "test"')['WSI'])
        train_set = load_data_info_from_list(train_list, gt_list, data_root_dir, label_type)

        if down_sample:
            train_normal = [train_set[i] for i in range(len(train_set)) if train_set[i][1] == 0]
            train_tumor = [train_set[i] for i in range(len(train_set)) if train_set[i][1] != 0]

            random.shuffle(train_normal)
            train_normal = train_normal[: len(train_tumor) // 3]
            train_set = train_normal + train_tumor

        valid_set = load_data_info_from_list(valid_list, gt_list, data_root_dir_2, label_type)
        test_set = load_data_info_from_list(test_list, gt_list, data_root_dir_2, label_type)
        return train_set, valid_set, test_set

    if nr_classes == 3:
        gt_train_local = {1: 4,  # "BN", #0
                          2: 4,  # "BN", #0
                          3: 0,  # "TW", #2
                          4: 1,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 4:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 5:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 8,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 6:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 3,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 5,  # "signet", #7
                          10: 5,  # "poorly", #7
                          11: 6  # "LVI", #ignore
                          }
    elif nr_classes == 8:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 3,  # "TM", #3
                          5: 4,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 7,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 10:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 8,  # "poorly", #7
                          11: 9  # "LVI", #ignore
                          }
    else:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 5,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }

    csv_her02 = '/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_cancer_wsi_1024_80_her01_split.csv'
    csv_addition = '/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition_PS1024_ano08_split.csv'

    data_her_root_dir = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step05_bright230_resize05'
    data_her_root_dir_2 = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step10_bright230_resize05'
    data_add_root_dir = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step05_bright230_resize05'
    data_add_root_dir_2 = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'

    train_set, valid_set, test_set = load_a_dataset(csv_her02, gt_train_local,data_her_root_dir, data_her_root_dir_2, label_type=label_type)
    train_set_add, valid_set_add, test_set_add = load_a_dataset(csv_addition, gt_train_local, data_add_root_dir, data_add_root_dir_2, down_sample=False, label_type=label_type)
    
    train_set += train_set_add
    valid_set += valid_set_add
    test_set += test_set_add

    return train_set, valid_set, test_set

def prepare_k19(label_type='caption'):
    data_root_dir = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/NCT-CRC-HE-100K/'
    json_dir = '/data1/trinh/code/DoIn/pycontrast/datasets/K19_9class_split.json'
    with open(json_dir) as json_file:
        data = json.load(json_file)

    train_set = data['train_set']
    valid_set = data['valid_set']
    test_set = data['test_set']
    train_set = [[data_root_dir + train_set[i][0], train_set[i][1]] for i in range(len(train_set))]
    valid_set = [[data_root_dir + valid_set[i][0], valid_set[i][1]] for i in range(len(valid_set))]
    test_set = [[data_root_dir + test_set[i][0], test_set[i][1]] for i in range(len(test_set))]

    mapping_dict = {
        0: 'adipole tissue.',
        1: 'background tissue.',
        2: 'debris tissue.',
        3: 'lymphocyte tissue.',
        4: 'debris tissue.',   # mucus -> debris (MUC->DEB)
        5: 'stroma tissue.',   # muscle -> stroma (MUS->STR)
        6: 'normal tissue.',
        7: 'stroma tissue.',
        8: 'tumor tissue.'
    }
    if label_type == 'caption':
        for i in range(len(train_set)):
            train_set[i][1] = mapping_dict[train_set[i][1]]
        
        for i in range(len(valid_set)):
            valid_set[i][1] = mapping_dict[valid_set[i][1]]
        
        for i in range(len(test_set)):
            test_set[i][1] = mapping_dict[test_set[i][1]]
    elif label_type == 'combine_dataset':
        for i in range(len(train_set)):
            train_set[i][1] += 10
        
        for i in range(len(valid_set)):
            valid_set[i][1] += 10
        
        for i in range(len(test_set)):
            test_set[i][1] += 10

    return train_set, valid_set, test_set
