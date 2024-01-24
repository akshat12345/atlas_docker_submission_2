import os,pathlib
import cv2 as cv
import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
from medpy.io import load,save
from settings import loader_settings
#import medpy

import torch
from torch import nn
from torch.autograd import Variable
import copy
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils


def process_slice(img,k):
    threshold_begin,threshold_end = 0.40 * img.shape[0] * img.shape[1],0.30 * img.shape[0] * img.shape[1]
    
    original = img.copy()
    original = np.clip(original,15,120)
    
    temp = np.zeros([192,192],dtype = np.float32)
    original_copy = original.copy()
    original_copy[original_copy <= 15] = 0
    
    if ((np.count_nonzero(original_copy) <= threshold_begin and k <= 20) or (np.count_nonzero(original_copy) <= threshold_end and k >= 160)):
        result_slice = temp
        
        all_zeros = True
    else:
        result_slice = original
        all_zeros = False
    
    return result_slice,all_zeros


def converttovoxel(image,intercept,slope):  
    image = image.astype(np.float32)
    
    if int(slope) != 1:
        image= slope * image        
    image += intercept
    return image


def preprocessing(data):
    print(data.shape)
    process_t1 = np.zeros([192,192,189],dtype = np.float32)
    for k in range(data.shape[2]):
        mrivox = data[:,:,k]
        result_slice,all_zeros = process_slice(mrivox,k)
        if np.count_nonzero(result_slice) != 0:
            mean = np.sum(result_slice)/np.count_nonzero(result_slice)
            std = result_slice[result_slice!=0].std()
            result_slice = (result_slice - mean)/std
        result_slice = cv.resize(result_slice,(192,192),interpolation=cv.INTER_CUBIC)
        if all_zeros:
            result_slice = np.zeros([192,192],dtype = np.float32)
        process_t1[:,:,k] = result_slice
#         plt.subplot(131),plt.imshow(result_slice,cmap=plt.cm.gray),plt.title('1')
#         temp = cv.resize(data[:,:,k],(192,192),interpolation=cv.INTER_CUBIC)
#         plt.subplot(132),plt.imshow(temp,cmap=plt.cm.gray),plt.title('2')
#         plt.show()
    
    print(process_t1.shape)
    process_t1 = np.expand_dims(process_t1,0)
    process_t1 = np.moveaxis(process_t1, 3,0)
    print(process_t1.shape)
    return process_t1

class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
        
    def process(self):
        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        
        print(file_list)
        device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
        model = smp.Unet(
            encoder_name="mit_b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            decoder_use_batchnorm = True,
            decoder_attention_type  = "scse",
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )
        model.load_state_dict(torch.load('./Unet-mit-b2-zeros-augmented2-d-10.pth',map_location=torch.device('cpu'))) #map_location=torch.device('cpu')
        model.to(device)

        for file in file_list:
            dat,hdr = load(file)
            im_shape = dat.shape
            print(im_shape)#_______________________
            print('before reshaping -> ' , dat.shape)
            dat = dat.reshape(*im_shape)
            print('after reshaping -> ', dat.shape)
            data = dat
            print(file.split('/')[-1])
            processed_img_np = preprocessing(data)

            y_pred = []
            for cnt in range(0,189,1):
                x = processed_img_np[cnt,:,:,:]
                x = np.expand_dims(x,0)
                x = torch.Tensor(x)
                x = torch.cat((x,)*3,dim=1) 
                #print(x.shape)
                if np.count_nonzero(x) != 0:
                    x = x.to(device)
                    with torch.set_grad_enabled(False):
                        pred = model(x).float().cpu().numpy()
                else:
                    pred = np.zeros((1,1,192,192),dtype = np.float32)
                y_pred.extend(pred)
            
            #y_pred.extend(np.zeros([13,1,192,192],dtype = np.float32))
            y_pred = np.array(y_pred)
            y_pred = np.squeeze(y_pred, axis=1)
            y_pred = np.transpose(y_pred, (1, 2, 0))
            print(y_pred.shape)
            
            thresh= 0.5
            
            final_ans = np.zeros([197,233,189],dtype = np.float32)
            for j in range(0,189):
                reshape_slice = cv.resize(y_pred[:,:,j], (233,197), interpolation=cv.INTER_CUBIC)
                final_ans[:,:,j] = reshape_slice

            final_ans_thresh = np.zeros([197,233,189],dtype = np.float32)
            
            final_ans_thresh[final_ans>thresh] = 1.0

            final_ans_thresh[final_ans<=thresh] = 0.0
            dat=final_ans_thresh
            print('before reshaping -> ' , dat.shape)
            dat = dat.reshape(*im_shape) 
            print('after reshaping -> ',dat.shape)
            print(os.path.basename(file))
            out_name = os.path.basename(file)
            out_filepath = os.path.join(out_path, out_name)
            print(f'=== saving {out_filepath} from {file} ===')
            save(dat, out_filepath, hdr=hdr)
            print("Done")
        return 



if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()
