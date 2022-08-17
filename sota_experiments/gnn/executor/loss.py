# Implementation of loss functions
# Use functional apis as much as possible
# Dependencies - Maybe some utils files
# Called by main, test, train val and probably metrics
import torch.nn.functional as F
import torch

def masked_loss(predictions, target):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    mask = target['num_relation']

    criterions = {}
    criterions['cr'] = F.cross_entropy
    criterions['lr'] = F.binary_cross_entropy_with_logits
    criterions['mr'] = F.binary_cross_entropy_with_logits

    loss = {}
    loss['loss_total'] = 0
    loss['cr'] = 0
    loss['mr'] = 0
    loss['lr'] = 0

    keys = ['cr', 'lr', 'mr']

    b_size = target['lr'].shape[0]

    for b in range(b_size):
        curr_num_rel = int(mask[b])
        
        temp_predictions = {}
        temp_targets = {}
        
        for k in keys:
            temp_predictions = predictions[k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            temp_loss = criterions[k](temp_predictions, temp_targets)
            
            loss['loss_total']+=(temp_loss/b_size)
            loss[k]+=(temp_loss/b_size)

    return loss



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

