import torchvision.models as models
import _pickle as cPickle

#reshaping weights to load to modified resnet18

resnet18 = models.resnet18(pretrained=True)

state_dict_new={}
for key in resnet18.state_dict():
    print(key, resnet18.state_dict()[key].shape)
    var=resnet18.state_dict()[key]
    if (key.find('conv') != -1 or key.find('downsample.0') != -1): 
        x = var.unsqueeze(0)
        x=x.permute(1,2,0,3,4)
        state_dict_new[key]=x
    else: 
         state_dict_new[key]=resnet18.state_dict()[key]
        
        
output = open('resnet18_weights.pkl', 'wb')
cPickle.dump(state_dict_new, output, 2)
output.close()