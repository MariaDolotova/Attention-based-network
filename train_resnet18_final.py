import numpy as np
import _pickle as cPickle
from .LoadDataPytorch import Dataset 
from torch.utils import data
import torch
import torch.optim as optim
import torch.nn as nn
import time
from vispy.io import imsave
import scipy

           
    
def train_resnet18_func():

 # Parameters
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 0}
    max_epochs = 100


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #dictionary - image_ids and correcponding labels
    input = open('AttentionBased/labels.pkl', 'rb')
    labels = cPickle.load(input)
    input.close()
 
    #image ids for training
    input = open('AttentionBased/list_train_ids.pkl', 'rb')
    list_train_ids = cPickle.load(input)
    input.close()
    
    #image ids for validation
    input = open('AttentionBased/list_val_ids.pkl', 'rb')
    list_val_ids = cPickle.load(input)
    input.close()
    
        
    partition={'train' : list_train_ids, 'validation' : list_val_ids}


    # Generators
    training_set = Dataset('train', partition['train'], labels)
    training_generator = data.dataloader.DataLoader(training_set, **params)

    validation_set = Dataset('val', partition['validation'], labels)
    validation_generator = data.dataloader.DataLoader(validation_set, **params)


    #creating the model - loading pretrained resnet18
   
    net=torch.load('AttentionBased/pretrained_resnet18.pth')
    cntr=0
    lt=10
    for child in net.children():
        cntr+=1
        if cntr < lt:
            for param in child.parameters():
                param.requires_grad = False

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)

    
    for epoch in range(max_epochs):
        train_log = open("AttentionBased/logs/train_log.txt","a") 
        val_log = open("AttentionBased/logs/val_log.txt","a") 
        train_log_str=""
        val_log_str=""
        # Training
        net.train()

        i=0
        running_loss = 0.0
        running_loss_val = 0.0
        for local_batch, local_labels in training_generator:

            i=i+1
            img_id=local_batch['img_id']
            
            time1 = time.time()

        # zero the parameter gradients
            optimizer.zero_grad()
            

        # forward + backward + optimize
            x1=np.asarray(local_batch['img_input']).astype(float)
            x1=torch.Tensor(x1)
            
            x2=local_batch['dim_input']
            
            
            y_tensor = torch.tensor(local_labels, dtype=torch.long, device=device)

            att_map, outputs = net(x1, x2)
            
            
            #   saving attention maps
            for f in range(64):
                ar=att_map[f, :, :]
                ar=ar.detach().numpy()
                ar=scipy.ndimage.zoom(ar, 16, order=0)
                filename='AttentionBased/maps/'+str(img_id)+'_attention_map_'+str(f)+'_epoch_'+str(epoch)+'.png'
                imsave(filename, ar)
                
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

          # saving statistics
            running_loss += loss.item()
            time2 = time.time()
            log_str='**************************training statistics: [%d, %5d, time: %.5f] loss: %.3f \n' % (epoch + 1, i, time2-time1, running_loss / i)
            print(log_str)
            train_log_str+=log_str
           
            
            
            

            # Validation
        net.eval()
        j=0
        running_loss_val = 0.0
        with torch.set_grad_enabled(False):
            for local_batch_val, local_labels_val in validation_generator:
                   
                j=j+1
                
                x1_val=np.asarray(local_batch_val['img_input']).astype(float)
                x1_val=torch.Tensor(x1_val)
                    
                x2_val=local_batch_val['dim_input']
                    
                    
                y_tensor_val = torch.tensor(local_labels_val, dtype=torch.long, device=device)

                atten_map, outputs_val = net(x1_val, x2_val)
                loss_val = criterion(outputs_val, y_tensor_val)

        
                # saving statistics
                running_loss_val += loss_val.item()
                log_str='***************************validation statistics: [%d, %5d] loss: %.3f \n' % (epoch + 1, j, running_loss_val/j)
                print(log_str)
                val_log_str+=log_str
        
        torch.save(net, 'AttentionBased/models/resnet18_weights_epoch_'+str(epoch)+'.pth')
        
        train_log.write(train_log_str)
        val_log.write(val_log_str)
        train_log.close()
        val_log.close()