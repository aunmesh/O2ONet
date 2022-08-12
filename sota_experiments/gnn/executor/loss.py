# Implementation of loss functions
# Use functional apis as much as possible
# Dependencies - Maybe some utils files
# Called by main, test, train val and probably metrics
import torch.nn.functional as F
import torch

def segmentation_loss(input, target, mask):
    """
    description:
    Calculate the segmentation loss

    args:
    input   :shape = [b,t,c_a]
    target  :shape = []
    
    returns:
    """
    batch_size, num_classes, time_steps = input.shape
    loss = {}
    
    loss['loss_total'] = 0

    for b in range(batch_size):
        temp_num_frames = mask[b]
        temp_input = input[b].transpose(0,1)[:temp_num_frames,:]
        temp_target = target[b,:temp_num_frames]
        loss['loss_total']+=F.cross_entropy(temp_input, temp_target)

    return loss


def OOI_masked_loss(input, target):
    """
    description:
    Calculate the segmentation loss

    args:
    input   :shape = [b,t,c_a]
    target  :shape = []
    
    returns:
    """

    raise NotImplementedError

