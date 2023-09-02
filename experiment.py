import argparse
parser = argparse.ArgumentParser(description = "Template")

parser.add_argument("-gpu", "--GPU_index", default = 0, type = int, help = "gpu index")
parser.add_argument("-thre", "--threshold", default = 0.3, type = float, help = "the threshold for the second possible class")
parser.add_argument("-model", "--model_name", default = "vgg", type = str, help = "the name of the model")
parser.add_argument("--negative", default=False, action="store_true")
parser.add_argument("-root", "--data_root", type = str, help = "the root of the dataset")

options = parser.parse_args()


import torch
import torchvision
import numpy as np
from torch import nn
from dataset import *
from utils import *

torch.manual_seed(0)
device=torch.device(f'cuda:{options.GPU_index}')


def experiment(samples, testset, model, index=0, mode='positive'):
    OriginalProb = []
    ContrastiveProb = []
    Prob = []
    from torchvision import transforms
    GaussianBlur = transforms.GaussianBlur(101, sigma=(10, 20))
    for (n, y1, y2) in tqdm(samples):
        image = testset[n][0].view(1,3,224,224).to(device)

        if index == 1:
            t = y1
        elif index == 2:
            t = y2

        with torch.no_grad():
            pred = model(image)[0]
            prob = torch.softmax(pred, dim = 0)


        blurred = GaussianBlur(image)

        p_blurred = image.detach().clone()
        n_blurred = image.detach().clone()

        gc_y = get_saliency(model, image, t, softmax=False)
        gc_p = get_saliency(model, image, t, softmax=True)

        y_images, p_images = equal_blur(image, gc_y, gc_p, mode=mode)


        with torch.no_grad():
            pred_p = model(p_images)[0]
            prob_p = torch.softmax(pred_p, dim = 0)
            pred_y = model(y_images)[0]
            prob_y = torch.softmax(pred_y, dim = 0)

        OriginalProb.append([torch.exp(pred[y1]).item(), torch.exp(pred[y2]).item()])
        ContrastiveProb.append([torch.exp(pred_p[y1]).item(), torch.exp(pred_p[y2]).item()])
        Prob.append([torch.exp(pred_y[y1]).item(), torch.exp(pred_y[y2]).item()])
    
    OriginalProb = np.array(OriginalProb)
    ContrastiveProb = np.array(ContrastiveProb)
    Prob = np.array(Prob)
    return OriginalProb, ContrastiveProb, Prob

if __name__ == '__main__':
    if options.model_name == 'vgg':
        model = torchvision.models.vgg16_bn(pretrained = False).to(device)
    elif model_name == 'alexnet':
        options.model = torchvision.models.alexnet(pretrained = False).to(device)
    model.classifier[6] = nn.Linear(4096, 200).to(device)
    model.load_state_dict(torch.load(f'model/{options.model_name}_CUB.pth'))

    model.eval()
    testset = CUB(options.data_root, normalization=True, train_test='test')

    samples = get_samples(testset,model,options.threshold)

    if not options.negative:
        OriginalProb, ContrastiveProb, Prob = experiment(samples, testset, model, index=0, mode='positive')
        mode == 'positive'
        desire = 'higher'
    else:
        OriginalProb, ContrastiveProb, Prob = experiment(samples, testset, model, index=0, mode='negative')
        mode == 'negative'
        desire = 'lower'

    print("The mode is %s, so the relative score should be the %s the better"%(mode, desire))
    print("Original\t\t r=%.4f"%((OriginalProb[:,0]/OriginalProb.sum(1)).mean()))
    print("Contrastive Blurred\t r=%.4f"%((ContrastiveProb[:,0]/ContrastiveProb.sum(1)).mean()))
    print("Blurred\t\t r=%.4f"%((Prob[:,0]/Prob.sum(1)).mean()))