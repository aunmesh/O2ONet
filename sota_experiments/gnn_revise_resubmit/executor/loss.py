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


def masked_loss_gpnn(predictions, target, criterions):
    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''

    # examine_data = {}
    # examine_data['predictions'] = predictions
    # examine_data['target'] = target 
    # examine_data['criterions'] = criterions
    
    # torch.save(examine_data, 'examine_data.pt')
    # if 1 == 1:
    #     return 'lol'
    mask = target['num_relation']

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
            
                
            temp_predictions = predictions['combined'][k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            if k == 'cr':
                temp_targets = torch.argmax(temp_targets, dim=-1)
                
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
    
    
    for k in keys:
        loss['loss_' + k] /= (1.0 * tot_num_rels)

    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))
    
    return loss



def masked_loss_mfurln(predictions, target, criterions):
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
    
    indeterminate_output = predictions['indeterminate_out']
    label_tensor = predictions['label_tensor']
    
    indeterminate_loss = F.binary_cross_entropy(indeterminate_output, label_tensor)
    
    device = predictions['label_tensor'].device
    
    num_neg = predictions['cr_neg'].shape[0]
    neg_cr_labels = torch.tensor([2]).expand(num_neg).to(device)
    neg_lr_labels = torch.tensor([0,0,0,0,1]).expand(num_neg,5).to(device).float()
    neg_mr_labels = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,1]).expand(num_neg,14).to(device).float()
    
    negative_loss_cr = F.cross_entropy(predictions['cr_neg'], neg_cr_labels)
    negative_loss_lr = F.binary_cross_entropy_with_logits(predictions['lr_neg'], neg_lr_labels)
    negative_loss_mr = F.binary_cross_entropy_with_logits(predictions['mr_neg'], neg_mr_labels)
    
    neg_loss_total = negative_loss_cr + negative_loss_lr + negative_loss_mr
    neg_loss_total/=3
    
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
            temp_predictions = predictions['combined'][k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
    
    
    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))
    loss['loss_total'] += neg_loss_total
    loss['loss_total'] += indeterminate_loss
    
    return loss




def masked_loss_graph_rcnn(predictions, target, criterions):

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
    
    
    # Implement the loss for the relationship proposal
    relatedness_score = predictions['rel_proposal']
    rel_proposal_loss = 0
    device = relatedness_score.device
    
    all_pairs = []
    for k1 in range(8):
        for k2 in range(8):
            all_pairs.append([k1, k2])
    
    all_pairs = torch.tensor(all_pairs, device=device)
    
    for b in range(b_size):
        curr_num_rel = int(mask[b])
        obj_pairs = target['object_pairs'][b, :curr_num_rel].to(device)
        
        neg_locs = (all_pairs == obj_pairs[0])
        
               
        for j in range(1,curr_num_rel):
            neg_locs += (all_pairs == obj_pairs[j])
        
        neg_locs = neg_locs[:,0] * neg_locs[:,1]
        neg_locs = torch.where(neg_locs==False)[0]
        
        neg_indices_0 = all_pairs[neg_locs, 0]
        neg_indices_1 = all_pairs[neg_locs, 1]
        
        neg_predictions = relatedness_score[b, neg_indices_0, neg_indices_1]
        target_tensor_0 = torch.zeros(neg_predictions.shape[0], device=device)
        
        # rel_proposal_loss+= F.binary_cross_entropy(neg_predictions, target_tensor_0)
                    
        
        temp_predictions_1 = relatedness_score[ b, obj_pairs[:,0], obj_pairs[:,1] ]
        temp_predictions_2 = relatedness_score[ b, obj_pairs[:,1], obj_pairs[:,0] ]
        
        target_tensor = torch.ones(curr_num_rel, device=device)
        
        rel_proposal_loss+= F.binary_cross_entropy(temp_predictions_1, target_tensor)
        rel_proposal_loss+= F.binary_cross_entropy(temp_predictions_2, target_tensor)
    
    rel_proposal_loss/=(2.0*b_size)
        
        
    
    for b in range(b_size):
        curr_num_rel = int(mask[b])
        tot_num_rels+=curr_num_rel
        
        temp_predictions = {}
        temp_targets = {}
        
        for k in keys:
            temp_predictions = predictions['combined'][k][b, :curr_num_rel, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
    
    
    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    
    loss['loss_total']/=( 1.0 * len(keys))
    loss['loss_total']+=rel_proposal_loss
    
    
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
                temp_predictions = predictions['combined'][k][lower_index:upper_index, :]
                temp_predictions = torch.log(temp_predictions + 1e-20)
                temp_targets = target[k][b, :curr_num_rel, :]
                
                temp_targets_indices = torch.max(temp_targets, dim=1)[1]
                temp_loss = criterions[k](temp_predictions, temp_targets_indices)
                loss['loss_' + k]+=temp_loss

                continue

            temp_predictions = predictions['combined'][k][lower_index:upper_index, :]
            temp_targets = target[k][b, :curr_num_rel, :]
            temp_loss = criterions[k](temp_predictions, temp_targets)
            loss['loss_' + k]+=temp_loss
        lower_index = upper_index
    
    
    # adding all losses together
    for k in keys:
        loss['loss_' + k] /= (1.0 * b_size)
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))
    print(loss['loss_total'])
    return loss

import numpy as np
def calc_bce(probab, target):

    num_vals = len(target)
    total_sum = 0

    for i in range(num_vals):
        temp_p = probab[i]
        temp_t = target[i]
        if temp_t == 1:
            temp_sum = -1 * np.log(temp_p)
        elif temp_t == 0:
            temp_sum = -1 * np.log(1 - temp_p)
        total_sum+=temp_sum

    # print(total_sum, total_sum/num_vals)
    return total_sum/num_vals

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
                    temp_predictions = temp_predictions + 1e-20
                    
                    temp_targets = target[k][b, :curr_num_rel, :]
                    temp_targets_indices = torch.max(temp_targets, dim=1)[1]
                    
                    if b==0:
                        pass
                        # print( "FLAG 1", temp_predictions )
                        # print(torch.max(temp_predictions, dim=1)[1].data, temp_targets_indices.data, "FLAG")
                    
                    temp_loss = criterions[k](temp_predictions, temp_targets_indices)
                    loss['loss_' + stream + '_stream_' + k] = temp_loss
                
                continue
            
            for stream in streams:
                
                temp_predictions = predictions[stream][k][lower_index:upper_index, :]
                temp_predictions = temp_predictions + 1e-20
                
                temp_targets = target[k][b, :curr_num_rel, :]
                temp_loss = criterions[k](temp_predictions, temp_targets)
                loss['loss_' + stream + '_stream_' + k] = temp_loss

                if b == 0 and k == 'lr':
                    temp_probabs = torch.sigmoid(temp_predictions.data[0, ...]).cpu()
                    # loss_val = temp_loss.detach().cpu().numpy()
                    loss_val_torch = criterions[k](temp_predictions[0, ...], temp_targets[0, ...] ).detach().cpu().numpy()
                    loss_val_self = calc_bce( temp_probabs, temp_targets[0, ...])
                    print(loss_val_torch, loss_val_self)
                    # print( temp_probabs.numpy(), 1.0 * (temp_probabs > 0.5).numpy(), temp_targets.data[0, ...].cpu().numpy(), loss_val )

        lower_index = upper_index
    
    # adding all losses together
    for k in keys:
        for stream in streams:
            loss['loss_total'] += loss['loss_' + stream + '_stream_' + k]

    loss['loss_total']/=( 1.0 * len(keys))

    return loss

def masked_loss_ican(predictions, target, criterions, config):

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
    
    # streams = config['streams']
    streams = ['combined']

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
                    temp_targets_indices = torch.max(temp_targets, dim=1)[1]
                    temp_loss = criterions[k](temp_predictions, temp_targets_indices)

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
            loss['loss_' + stream + '_stream_' + k] /= (1.0 * b_size)
            loss['loss_total'] += loss['loss_' + stream + '_stream_' + k]

    loss['loss_total']/=( 1.0 * len(keys) * len(streams))
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