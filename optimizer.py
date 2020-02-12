import torch.optim

def sgd(weight, lr, momentum=0.9, lr_decay=0):
    return torch.optim.SGD(weight, lr, momentum)

OPTIM = {
    'SGD': sgd,
}