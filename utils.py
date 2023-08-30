import torch
import torchvision

def get_n_params(model):
    n_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_param += torch.tensor(param.shape).prod()
    print(f'Then number of parameters is {n_param}.')
    

def normalize(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(device)
    xx[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0] 
    xx[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1] 
    xx[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2] 
    return xx

def denorm(x):
    device = x.device
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(device)
    
    if len(xx.shape) == 4:
        xx[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
        xx[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
        xx[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    elif len(xx.shape) == 3:
        xx[0, :, :] = x[0, :, :] * std[0] + mean[0]
        xx[1, :, :] = x[1, :, :] * std[1] + mean[1]
        xx[2, :, :] = x[2, :, :] * std[2] + mean[2]
    return xx

def plot_tensor_image(image):
    toshow = image.detach().clone().cpu()
    toshow = toshow.squeeze()
    if toshow.max() > 1:
        toshow = denorm(toshow)
    
    if len(toshow.shape) == 4:
        print("Plotting the entire batch:")
        toshow = torchvision.utils.make_grid(toshow)
        figsize = (20,20)
    else:
        print("Plotting the single image")
        figsize = (5,5)
    
    plt.figure(figsize=figsize)
    plt.imshow(toshow.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.show()