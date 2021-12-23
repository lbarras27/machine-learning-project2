import cv2
import os, sys
import numpy as np
import torch

def loadDataset(path, color = True):
    imgs_src = os.listdir(path)

    imgs = []
    for img_name in imgs_src:
        if color:
            img = cv2.imread(path + "/" + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(path + "/" + img_name, cv2.IMREAD_GRAYSCALE)

        imgs.append(img)

    imgs = np.array(imgs)
    
    return imgs

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotateDatasetImages(data_train, data_label, angle):
    imgs_train_rotate = []
    imgs_label_rotate = []
    for i in range(0, len(data_train)):
        img_train_rot = rotate_image(data_train[i], angle)
        img_label_rot = rotate_image(data_label[i], angle)   
        imgs_train_rotate.append(img_train_rot)
        imgs_label_rotate.append(img_label_rot)
    
    imgs_train_rotate = np.array(imgs_train_rotate)
    imgs_label_rotate = np.array(imgs_label_rotate)
    
    return imgs_train_rotate, imgs_label_rotate
    
def flipDatasetImages(data_train, data_label, direction='vertical'):
    d = 0
    if direction == 'vertical':
        d = 1
    else:
        d = 0
        
    imgs_train_flip = []
    imgs_label_flip = []
    for i in range(0, len(data_train)):
        img_train_fl = cv2.flip(data_train[i], d)
        img_label_fl = cv2.flip(data_label[i], d)    
        imgs_train_flip.append(img_train_fl)
        imgs_label_flip.append(img_label_fl)
    
    imgs_train_flip = np.array(imgs_train_flip)
    imgs_label_flip = np.array(imgs_label_flip)
    
    return imgs_train_flip, imgs_label_flip

def saveImages(img, path):
    for i in range(0, img.shape[0]):
        cv2.imwrite(path+'/satImage_'+ '%.3d' % (i+1) + '.png', img[i])
        

def resizeTrainSetLikeTestSet(data_train, data_label, path_directory):
    train_resize = []
    label_resize = []
    for i in range(0, data_train.shape[0]):
        resize_t = cv2.resize(data_train[i], (600, 600))
        resize_t = cv2.copyMakeBorder(resize_t, 4, 4, 4, 4, cv2.BORDER_REFLECT) 
        resize_l = cv2.resize(data_label[i], (600, 600))
        resize_l = cv2.copyMakeBorder(resize_l, 4, 4, 4, 4, cv2.BORDER_REFLECT) 
        
        train_resize.append(resize_t)
        label_resize.append(resize_l)

    train_resize = np.array(train_resize)
    label_resize = np.array(label_resize)
    
    saveImages(train_resize, path_directory)
    saveImages(label_resize, path_directory)
    
    
def splitDataTrainVal(data_train, data_label, split, shuffle=True):
    if shuffle:
        idx = torch.randperm(data_train.size(0))
    else:
        idx = torch.arange(data_train.size(0))
        
    train_data = data_train[idx[:]]
    train_label = data_label[idx[:]]

    val_data = train_data[split:]
    train_data = train_data[0:split]
    val_label = train_label[split:]
    train_label = train_label[0:split]
    
    return train_data, train_label, val_data, val_label
    
    
def parseInputs(imgs):
    imgs = torch.from_numpy(imgs).float()
    #imgs = imgs / 255
    imgs = (imgs - imgs.mean()) / imgs.std()
    imgs = imgs.permute(0, 3, 1, 2)
    
    return imgs
    
def parseLabels(imgs):
    imgs[imgs > 128] = 255
    imgs[imgs <= 128] = 0
    imgs = imgs / 255
    
    imgs = torch.from_numpy(imgs).float()
    #imgs = torch.from_numpy(imgs).long()
    imgs = imgs[None, ...]
    imgs = imgs.permute(1, 0, 2, 3)
    
    return imgs
    

def parseTestSet(path, patch_size=304):
    data_test = loadDataset(path)
    data_test = torch.from_numpy(data_test)
    data_test = data_test / 255
    data_test = data_test.permute(0, 3, 1, 2)
    patchs_test = data_test.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patchs_test = patchs_test.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)
    
    return patchs_test
    
    
def convert_patch_to_mask(output_path, patchs_test, model, device):
    for m in range(0, 50):
        mask = torch.zeros(304, 304)
        res, patch = model(patchs_test[m*4:(m+1)*4].to(device))
        patch[patch < 0.5] = 0
        patch[patch > 0.5] = 255
        imf = torch.zeros(608, 608)
        for k in range(2):
            for l in range(2):
                for i in range(19):
                    for j in range(19):
                        mask[i*16:(i+1)*16, j*16:(j+1)*16] = torch.full((16, 16), patch[k*2+l, 0, i,j].item())
                imf[k*304:(k+1)*304, l*304:(l+1)*304] = mask
        
        cv2.imwrite(output_path+'/satImage_'+ '%.3d' % (m+1) + '.png', imf.cpu().detach().numpy())
    

def generate_groundtruth_test_set(patchs_test, model, device, output_path='results/resC'):
    num_patch = 4 # (2x2)
    for k in range(0, 50):
        res, _ = model(patchs_test[k*num_patch:(k+1)*num_patch].to(device))
        
        res[res < 0.5] = 0
        res[res >= 0.5] = 255

        imf = torch.zeros(608, 608)
        for i in range(0, num_patch//2):
            for j in range(0, num_patch//2):
                imf[i*304:(i+1)*304, j*304:(j+1)*304] = res[i*(num_patch//2)+j, 0]

        cv2.imwrite(output_path+'/satImage_'+ '%.3d' % (k+1) + '.png', imf.cpu().detach().numpy())
    
    
def applyNetworkOnTestSetAndsaveResults(model, path_test, path_result):   
    for i in range(1, 51):
        path_test = path_test + "/test_" + str(i)
        data_test = util.loadDataset(path_test)
        
        data_test = util.parseInputs(data_test)

        result = model(data_test.to(device))
        result[result < 0.5] = 0
        result[result >= 0.5] = 255
        cv2.imwrite(path_result+'/satImage_'+ '%.3d' % i + '.png', result[0, 0, ...].cpu().detach().numpy())