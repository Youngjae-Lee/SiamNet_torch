import torch.nn.functional as F


def BCELogit_Loss(score_map, labels):
    labels = labels.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(score_map, labels[:,:,:,:,0],
                                              weight=labels[:,:,:,:,1],
                                              reduction='mean')
    return loss
