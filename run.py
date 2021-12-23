import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.io
from torch.utils.data import DataLoader
import cv2
import os
import argparse
import train2
import util
import mask_to_submission as mts
import network2
from augmented_dataset import AugmentedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)



path_model = "models/Unet_950epoch.pt"
train_path = "dataset/training/images"
label_path = "dataset/training/groundtruth"
test_path = "dataset/test_set/test_imgs"

parser = argparse.ArgumentParser()
parser.add_argument('--train', action="store_true", help="Use mode training if not specify use test mode")
parser.add_argument('--pretrain', action="store_true", help="Use my training model")
parser.add_argument('-e', '--epoch', type=int, help="The number of iteration of the full data")
parser.add_argument('--lr', type=float, help="The learning_rate")
parser.add_argument('-o', '--output', help="Directory that will contain the results output images ")
parser.add_argument('-b', '--batch_size', type=int, help="The batch size")
args = parser.parse_args()

out_path = args.output if args.output else "results/res"
train_mode = args.train
pretrained = args.pretrain

if train_mode:
    epoch = args.epoch if args.epoch else 500
    lr = args.lr if args.lr else 0.001
    batch_size = args.batch_size if args.batch_size else 20


    augmented_dataset = AugmentedDataset(
        file_path_data=train_path,
        file_path_label=label_path
    )
    
    dataloader_train = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

model = network2.Unet()
model = model.to(device)
if pretrained:
    print("Load pretrain model: " + path_model)
    model.load_state_dict(torch.load(path_model))

if train_mode:
    print("*******Starting training*********")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train2.train(dataloader_train, model, optimizer, epoch, device)
    print("*******Training completed*********")
    print("Saving model in /models")
    torch.save(model.state_dict(), "models/new_model")

print("Generating results in " + out_path)
patchs_test = util.parseTestSet(test_path)
util.generate_groundtruth_test_set(patchs_test, model, device, out_path)
print("results generated")

print("Creating submission file")
submission_filename = 'submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = out_path+'/satImage_' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)
mts.masks_to_submission(submission_filename, *image_filenames)
print("Submission file created")
    
