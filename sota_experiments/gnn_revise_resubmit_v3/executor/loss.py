# Implementation of loss functions
# Use functional apis as much as possible
# Dependencies - Maybe some utils files
# Called by main, test, train val and probably metrics
import torch.nn.functional as F
import torch



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






def masked_loss_squat(predictions, target, criterions):
    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
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


    esm_masks = predictions['combined']['masks']

    pairs = target['object_pairs']
    num_rel = target['num_relation']
    tot_num_rels = 0    

    # collected_masks = []
    # collected_gts = []
    
    rel_proposal_loss = 0
    target_tensor = torch.ones(1, device=esm_masks[0].device).squeeze()
    
    for b in range(b_size):
    
        curr_num_rel = int(num_rel[b])
        tot_num_rels += curr_num_rel
            
        for i in range(curr_num_rel):

            ind0, ind1 = pairs[b, i, 0], pairs[b, i, 1]
            
            val1 = esm_masks[0][b, ind0, ind1]
            val2 = esm_masks[1][b, ind0, ind1]
            val3 = esm_masks[2][b, ind0, ind1]
            
            rel_proposal_loss+= F.binary_cross_entropy_with_logits(val1, target_tensor)
            rel_proposal_loss+= F.binary_cross_entropy_with_logits(val2, target_tensor)
            rel_proposal_loss+= F.binary_cross_entropy_with_logits(val3, target_tensor)
    
    
    rel_proposal_loss/=(3.0*tot_num_rels)    
    loss['esm_loss'] = rel_proposal_loss
    
    # adding all losses together
    for k in keys:
        loss['loss_total'] += loss['loss_' + k]

    loss['loss_total']/=( 1.0 * len(keys))
    loss['loss_total'] += loss['esm_loss']
    
    return loss










def masked_loss_graph_rcnn(predictions, target, criterions):

    '''
    predictions : list of dimension [b_size, max_num_obj_pairs, num_classes]
    target      : list of dimension [b_size, max_num_obj_pairs, num_classes]
                  they correspond with the predictions
    mask        : A mask of dimension [b_size, max_num_obj_pairs]
    '''



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

