import torch
import torch.nn as nn
from sklearn.metrics import f1_score

####### For the first model #############
#def train2(model, e, input, label, val_input, val_label, optimizer, mini_batch_size):
#    model.train()
#    sum_loss = 0
#    criterion = nn.BCELoss()
#    for b in range(0, input.size(0), mini_batch_size):
#        optimizer.zero_grad()
#        output = model(input.narrow(0, b, mini_batch_size).to(device))
        
#        loss = criterion(output, label.narrow(0, b, mini_batch_size).to(device))

#        sum_loss = sum_loss + loss.item()
        
#        loss.backward()
#        optimizer.step()

#    print(e, sum_loss, loss)
    

"""
    train the model with data that is generate with the dataloader and optimize with the 
    optimizer get in paramater.
    
    @param dataloader: the dataloader that generate the image with its corresponding groundtruth
    @param model: the model to train 
    @param optimizer: the optimizer used during the training
    @param epoch: the number of iterations
    @param device: the corresponding device (cpu or cuda)
    @param verbose: if True, print the loss error
"""
def train(dataloader, model, optimizer, epoch, device, lambda1=1, lambda2=1, verbose=True, mode_patch=True):
    model.train()
    criterion = nn.BCELoss()
    
    for e in range(epoch):
        sum_loss = 0
        for data, label in dataloader:
            optimizer.zero_grad()
            
            if mode_patch:
                output, out_patch = model(data.to(device))
            else:
                output = model(data.to(device))
            
            loss_mask = criterion(output, label.to(device))
            
            if mode_patch:
                # transform the groundtruth in patches of 16x16 pixels
                batch_label = label[:, 0]
                patch_label = batch_label.unfold(1, 16, 16).unfold(2, 16, 16).mean((3, 4))[:, None]
                patch_label[patch_label > 0.25] = 1
                patch_label[patch_label <= 0.25] = 0
            
                # objective loss: compare the 16x16 patches
                loss_batch = criterion(out_patch, patch_label.to(device))
            
            if mode_patch:
                loss = lambda1 * loss_mask + lambda2 * loss_batch
            else:
                loss = loss_mask
                
            sum_loss = sum_loss + loss.item()
            
            loss.backward()
            optimizer.step()
        
        if verbose:
            print(e, sum_loss)
    
    
"""
    Compute the accuracy of the model with the data generate with dataloader.
    
    @param model: The model we want to compute the accuracy
    @param dataloader: the dataloader that generate the image with its corresponding groundtruth
    @param device: the corresponding device (cpu or cuda)
    
    @return the accuracy of the model
""" 
def accuracy(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        sum = 0
        num_batch = 0
        for data, label in dataloader:
            res, patch = model(data.to(device))
            res[res < 0.5] = 0
            res[res >= 0.5] = 1
            label[label < 0.5] = 0
            label[label >= 0.5] = 1
            diff = 1 - (((res.cpu() - label).abs()).sum()/(data.shape[2]*data.shape[3]*len(data)))
            sum = sum + diff
            num_batch += 1
            
    sum = sum / num_batch

    return sum
    

"""
    Compute the f1 score of the model with the data generate with dataloader.
    
    @param model: The model we want to compute the f1 score
    @param dataloader: the dataloader that generate the image with its corresponding groundtruth
    @param device: the corresponding device (cpu or cuda)
    
    @return the f1 score of the model
"""
def f1score(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        score = 0
        num_batch = 0
        for data, label in dataloader:
            res, patch = model(data.to(device))
            res[res < 0.5] = 0
            res[res >= 0.5] = 1
            label[label < 0.5] = 0
            label[label >= 0.5] = 1
            score += f1_score(label.view(-1).numpy(), res.view(-1).cpu().detach().numpy())
            num_batch += 1
    score /= num_batch
    
    return score
    