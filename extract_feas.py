import glob
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import os
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
import argparse
import pandas as pd
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def main(args):
    file_path = args.csv_path
    fls = pd.read_csv(file_path)
    paths = fls.iloc[0:]['name']
    labels = fls.iloc[0:]['label']
    # path = 'E:\\BVI-VFI-database\\code\\frames\\540p\\30Hz\\BVI_guitar_focus_960x540_30fps_Average'
    # path = path + '\\frame_*'
    # label = [1,0,0,0,0]
    image_size = (512, 512) # (256,256)

    
    

    def extract_features(path):
        files = glob.glob(path)
        grouped_files = [files[i:i + 12] for i in range(0, len(files) - 11, 12)]
        feature = np.empty((0, 1024))
        for group in grouped_files:
            images = torch.empty([0, 512, 512])#.to(args.device)
            images_2 = torch.empty([0, 512, 512])#.to(args.device)
            for file in group:
                image_orig = Image.open(file)
                div_factor = np.random.choice([1,2],1)[0]
                image_2 = ResizeCrop(image_orig, image_size, div_factor)
                image = ResizeCrop(image_orig, image_size, 3 - div_factor)

                # transform to tensor
                image = transforms.ToTensor()(image)#.unsqueeze(0).cuda()
                image_2 = transforms.ToTensor()(image_2)#.unsqueeze(0).cuda()

                images = torch.cat((images, image), dim=0)
                images_2 = torch.cat((images_2, image_2), dim=0)

            images = images.unsqueeze(0).cuda()
            images_2 = images_2.unsqueeze(0).cuda()

            # load CONTRIQUE Model
            encoder = get_network('resnet50', pretrained=False)
            model = CONTRIQUE_model(args, encoder, 512)
            model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
            model = model.to(args.device)

            # extract features
            model.eval()
            with torch.no_grad():
                _,_, _, _, model_feat, model_feat_2, _, _ = model(images, images_2)
            feat = np.hstack((model_feat.detach().cpu().numpy(),\
                        model_feat_2.detach().cpu().numpy()))
            
            
            feature = np.vstack((feat, feature))

        print(feature.shape)
        print('here here')
        
        return feature


    BVI_fea_1080_256 = {}
    cnt = 0
    for path in paths:
        if '1920x1080' in path:
            path1 = path + '\\frame_*'
            feat = extract_features(path1)
            # print(feat)
            BVI_fea_1080_256[path] = feat
            cnt += 1
            print(cnt)

    # save features
    path = './bvi_1080_256.pkl'
    with open(path, 'wb') as f:
        pickle.dump(BVI_fea_1080_256, f)
    # np.savez(args.feature_save_path, BVI_fea)
    print('1080 Done')


    BVI_fea_540_256 = {}
    cnt = 0
    for path in paths:
        if '960x540' in path:
            path1 = path + '\\frame_*'
            feat = extract_features(path1)
            # print(feat)
            BVI_fea_540_256[path] = feat
            cnt += 1
            print(cnt)

    # save features
    path = './bvi_540_256.pkl'
    with open(path, 'wb') as f:
        pickle.dump(BVI_fea_540_256, f)
    # np.savez(args.feature_save_path, BVI_fea)
    print('540 Done')


    BVI_fea_2160_256 = {}
    cnt = 0
    for path in paths:
        if '3840x2160' in path:
            path1 = path + '\\frame_*'
            feat = extract_features(path1)
            # print(feat)
            BVI_fea_2160_256[path] = feat
            cnt += 1
            print(cnt)

    # save features
    path = './bvi_2160_256.pkl'
    with open(path, 'wb') as f:
        pickle.dump(BVI_fea_2160_256, f)
    # np.savez(args.feature_save_path, BVI_fea)
    print('2160 Done')

    # BVI_fea_30 = {}
    # cnt = 0
    # for path in paths:
    #     if '30fps' in path:
    #         path1 = path + '\\frame_*'
    #         feat = extract_features(path1)
    #         # print(feat)
    #         BVI_fea_30[path] = feat
    #         cnt += 1
    #         print(cnt)

        
    # # save features
    # path = './bvi_30.pkl'
    # with open(path, 'wb') as f:
    #     pickle.dump(BVI_fea_30, f)
    # # np.savez(args.feature_save_path, BVI_fea)
    # print('30fps Done')

    # BVI_fea_60 = {}
    # cnt = 0
    # for path in paths:
    #     if '60fps' in path:
    #         path1 = path + '\\frame_*'
    #         feat = extract_features(path1)
    #         # print(feat)
    #         BVI_fea_60[path] = feat
    #         cnt += 1
    #         print(cnt)

        
    # # save features
    # path = './bvi_60.pkl'
    # with open(path, 'wb') as f:
    #     pickle.dump(BVI_fea_60, f)
    # # np.savez(args.feature_save_path, BVI_fea)
    # print('60fps Done')

    # BVI_fea_120 = {}
    # cnt = 0
    # for path in paths:
    #     if '120fps' in path:
    #         path1 = path + '\\frame_*'
    #         feat = extract_features(path1)
    #         # print(feat)
    #         BVI_fea_120[path] = feat
    #         cnt += 1
    #         print(cnt)

        
    # # save features
    # path = './bvi_120.pkl'
    # with open(path, 'wb') as f:
    #     pickle.dump(BVI_fea_120, f)
    # # np.savez(args.feature_save_path, BVI_fea)
    # print('120fps Done')


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_path', type=str, \
                        default='BVI_VFI_files.csv', \
                        help='list of filenames and labels of images', metavar='')
    parser.add_argument('--model_path', type=str, \
                        default='checkpoints/checkpoint_25.tar', \
                        help='Path to trained CONTRIQUE model', metavar='')
    parser.add_argument('--feature_save_path', type=str, \
                        default='features.npz', \
                        help='Path to save_features', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)