# Implementation of loss functions
# Use functional apis as much as possible
# Dependencies - Maybe some utils files
# Called by main, test, train val and probably metrics
import torch.nn.functional as F
import torch

def masked_loss_old(predictions, target, criterions):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    mask = target['num_relation']

    # criterions = {}
    # criterions['cr'] = F.cross_entropy
    # criterions['lr'] = F.binary_cross_entropy_with_logits
    # criterions['mr'] = F.binary_cross_entropy_with_logits

    loss = {}
    loss['loss_total'] = 0
    loss['loss_cr'] = 0
    loss['loss_mr'] = 0
    loss['loss_lr'] = 0

    keys = ['cr', 'lr', 'mr']
    
    b_size = target['lr'].shape[0]
    
    predictions_temp = {}
    target_temp = {}
    
    for k in keys:
        predictions_temp[k] = torch.cat([ predictions[k][i, :mask[i], :] for i in range(b_size) ], 0)
        target_temp[k] = torch.cat([ target[k][i, :mask[i], :] for i in range(b_size) ], 0)
            
    for k in keys:

        temp_loss = criterions[k](predictions_temp[k], target_temp[k])
        loss['loss_total']+=(temp_loss/b_size)
        loss['loss_' + k]+=(temp_loss/b_size)

    # print("DEBUG")
    # show = 10
    
    # for k in keys:
        
    #     print(" Check values for ", k)
        
    #     if k == 'cr':
    #         print( ( (F.softmax(predictions_temp[k]) >= 0.5) * 1.0)[:show, :])
        
    #     else:
    #         print(( ( F.sigmoid(predictions_temp[k]) >= 0.5) * 1.0)[:show,:])

    #     print(target_temp[k][:show,:])
    # print("###################")

    return loss


def masked_loss(predictions, target, criterions):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    mask = target['num_relation']

    # criterions = {}
    # criterions['cr'] = F.cross_entropy
    # criterions['lr'] = F.binary_cross_entropy_with_logits
    # criterions['mr'] = F.binary_cross_entropy_with_logits

    loss = {}
    loss['loss_total'] = 0
    loss['loss_cr'] = 0
    loss['loss_mr'] = 0
    loss['loss_lr'] = 0

    keys = ['cr', 'lr', 'mr']
    # norm_vals = {'cr' : 1, 'lr': 5, 'mr': 14}        

    b_size = target['lr'].shape[0]
    tot_num_rels = 0
    
    for b in range(b_size):
        curr_num_rel = int(mask[b])
        tot_num_rels+=curr_num_rel
        
        temp_predictions = {}
        temp_targets = {}
        
        for k in keys:
            temp_predictions = predictions[k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
    
    
    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))

    return loss


def masked_loss_encouraging_positive(predictions, target, criterions, pcp_hyperparameter):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    pcp_hyperparameter: A hyperparameter for positive content penalty
    '''

    mask = target['num_relation']

    loss = {}
    loss['loss_total'] = 0
    loss['loss_cr'] = 0
    loss['loss_mr'] = 0
    loss['loss_lr'] = 0

    keys = ['cr', 'lr', 'mr']

    b_size = target['lr'].shape[0]
    tot_num_rels = 0
    
    positive_content_penalty = {}
    
    
    for b in range(b_size):
        curr_num_rel = int(mask[b])
        tot_num_rels+=curr_num_rel
        
        temp_predictions = {}
        temp_targets = {}
        
        for k in keys:
            temp_predictions = predictions[k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            temp_loss = criterions[k](temp_predictions, temp_targets)
            
            if k =='lr' or k== 'mr':
                positive_content_penalty[k] = torch.mean(1 - torch.sigmoid(temp_predictions))
             
            loss['loss_' + k]+=temp_loss

    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))
    
    for k in ['lr', 'mr']:
        loss['loss_total']+=(pcp_hyperparameter * positive_content_penalty[k] * 0.5)

    return loss


def masked_loss_vsgnet(predictions, target, criterions):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    mask = target['num_relation']

    # criterions = {}
    # criterions['cr'] = F.cross_entropy
    # criterions['lr'] = F.binary_cross_entropy_with_logits
    # criterions['mr'] = F.binary_cross_entropy_with_logits

    loss = {}
    loss['loss_total'] = 0
    loss['loss_cr'] = 0
    loss['loss_mr'] = 0
    loss['loss_lr'] = 0

    keys = [ 'lr', 'mr', 'cr']
    # norm_vals = {'cr' : 1, 'lr': 5, 'mr': 14}        

    b_size = target['lr'].shape[0]
    
    lower_index = 0
    upper_index = 0

    for b in range(b_size):
        curr_num_rel = int(mask[b])
        upper_index += curr_num_rel
        
        for k in keys:
            
            if k == 'cr':
                temp_predictions = predictions[k][lower_index:upper_index, :]
                temp_predictions = torch.log(temp_predictions + 1e-20)
                temp_targets = target[k][b, :curr_num_rel, :]
                temp_loss = criterions[k](temp_predictions, temp_targets)
                loss['loss_' + k]+=temp_loss

                continue

            temp_predictions = predictions[k][lower_index:upper_index, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
        lower_index = upper_index
    
    
    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))

    return loss



def masked_loss_drg(predictions, target, criterions, config):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    mask = target['num_relation']

    # criterions = {}
    # criterions['cr'] = F.cross_entropy
    # criterions['lr'] = F.binary_cross_entropy_with_logits
    # criterions['mr'] = F.binary_cross_entropy_with_logits

    loss = {}
    loss['loss_total'] = 0
    loss['loss_cr'] = 0
    loss['loss_mr'] = 0
    loss['loss_lr'] = 0

    keys = [ 'lr', 'mr', 'cr']
    streams = config['streams']

    b_size = target['lr'].shape[0]
    
    lower_index = 0
    upper_index = 0

    for b in range(b_size):
        curr_num_rel = int(mask[b])
        upper_index += curr_num_rel
        
        for k in keys:
            
            if k == 'cr':
                for stream in streams:
                    temp_predictions = predictions[stream][k][lower_index:upper_index, :]
                    temp_predictions = torch.log(temp_predictions + 1e-20)
                    temp_targets = target[k][b, :curr_num_rel, :]
                    temp_loss = criterions[k](temp_predictions, temp_targets)
                    loss['loss_' + stream + '_stream_' + k] = temp_loss
                continue
            
            for stream in streams:
                temp_predictions = predictions[stream][k][lower_index:upper_index, :]
                temp_targets = target[k][b, :curr_num_rel, :]
                temp_loss = criterions[k](temp_predictions, temp_targets)
                loss['loss_' + stream + '_stream_' + k] = temp_loss

        lower_index = upper_index
    
    # adding all losses together
    for k in keys:
        for stream in streams:
            loss['loss_total'] += loss['loss_' + stream + '_stream_' + k]

    loss['loss_total']/=( 1.0 * len(keys))

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