# Implementations for all the metrics
import torch
import torch.nn.functional as F
import torchmetrics as tm
import copy
import torch.nn.functional as F

class metric_tracker_action_recog:

    def __init__(self, config, mode='train'):

        '''
        # Make Metric Objects
        # Needed metrics:

        average precision
        mAP
        confusion matrix 
        '''

        self.config = config
        device = self.config['device']

        self.action_categories = [ 'AssembleCabinet', 'ChangeCarTire', 'FuelCar', 'InstallBicycleRack', 'InstallShowerHead', 'ParkParallel',
                             'PolishCar', 'ReplaceBatteryOnTVControl', 'ReplaceDoorKnob', 'ReplaceToiletSeat', 'UseJack']

        self.action_classes = len(self.action_categories)
        self.mode = mode + '_'

        self.metrics = {}
        
        self.metrics[self.mode +'Accuracy' ] = tm.Accuracy(num_classes=self.action_classes, average='macro').to(device)

    def calc_metrics(self, pred, gt):

        result = {}
        
        pred = F.softmax(pred['action_index_logit'], dim=1)

        # Calculating Accuracy metrics
        result[self.mode + 'Accuracy'] = self.metrics[self.mode + 'Accuracy']( pred, gt['action_index'] )
        
        # Metric for deciding whether to save the model or not
        result['model_score'] = result[self.mode + 'Accuracy']

        return result
    
    def aggregate_metrics(self):

        result = {}
        
        # Calculating mAP metrics
        result[self.mode + 'Accuracy'] = self.metrics[self.mode + 'Accuracy'].compute()
        result['model_score'] = result[self.mode + 'Accuracy']
        
        return result


    def reset_metrics(self):

        self.metrics[self.mode + 'Accuracy'].reset()


class metric_tracker:

    def __init__(self, config, mode='train'):

        '''
        # Make Metric Objects
        # Needed metrics:

        average precision
        mAP
        confusion matrix 
        '''

        self.config = config
        device = self.config['device']

        self.cr_relations = ['Contact', 'No Contact', 'None of these']

        self.lr_relations = ['Below/Above', 'Behind/Front', 'Left/Right', 'Inside', 'None of these']

        self.mr_relations = ['Holding', 'Carrying', 'Adjusting', 'Rubbing', 'Sliding', 'Rotating',
                       'Twisting', 'Raising', 'Lowering', 'Penetrating', 'Moving Toward', 
                       'Moving Away', 'Negligible Relative Motion', 'None of these']

        self.cr_classes = 3
        self.lr_classes = 5
        self.mr_classes = 14
        self.mode = mode + '_'

        self.metrics = {}
        
        self.relation_keys = ['mr', 'lr', 'cr']
            
        self.metrics[self.mode +'mAP_lr' ] = tm.AveragePrecision(num_classes=self.lr_classes, average='macro').to(device)
        self.metrics[self.mode + 'mAP_mr'] = tm.AveragePrecision(num_classes=self.mr_classes, average='macro').to(device)
        self.metrics[self.mode + 'mAP_cr'] = tm.AveragePrecision(num_classes=self.cr_classes, average='macro').to(device)

        self.metrics[self.mode + 'AP_lr'] = tm.AveragePrecision(num_classes=self.lr_classes, average=None).to(device)
        self.metrics[self.mode + 'AP_mr'] = tm.AveragePrecision(num_classes=self.mr_classes, average=None).to(device)
        self.metrics[self.mode + 'AP_cr'] = tm.AveragePrecision(num_classes=self.cr_classes, average=None).to(device)

    # Accumulates all the tensors using mask in one matrix
    def make_tensor(self, predictions, gt, mask):
        '''
        This function makes a combined tensor for ground truth and predictions using the mask

            pred: pred is a dictionary of predictions
            gt  : gt is dictionary of ground truth labels
            mask: mask is the places which should be considered 
        '''
        out_gt = {}
        out_pred = {}

        for k in self.relation_keys:
            out_pred[k] = []
            out_gt[k] = []

        # num_relations in different batch elements
        num_relations = torch.split(mask, 1)

        for b, n in enumerate(num_relations):

            temp_num_relation = int(torch.sum(n))
            # temp_num_relation = int(n)

            for k in self.relation_keys:
                out_gt[k].append( gt[k][b,:temp_num_relation,:] )
                out_pred[k].append( predictions[k][b,:temp_num_relation,:] )

        for k in self.relation_keys:
            out_gt[k] = torch.cat(out_gt[k], 0)
            out_pred[k] = torch.cat(out_pred[k], 0)

        return out_pred, out_gt


    # Accumulates all the tensors using mask in one matrix
    def make_tensor_vsgnet(self, predictions, gt, mask):
        '''
        This function makes a combined tensor for ground truth and predictions using the mask

            pred: pred is a dictionary of predictions
            gt  : gt is dictionary of ground truth labels
            mask: mask is the places which should be considered 
        '''
        
        out_gt = {}

        for k in self.relation_keys:
            out_gt[k] = []

        # num_relations in different batch elements
        num_relations = torch.sum(mask)
        batch_size = mask.shape[0]
        
        lower_index = 0
        upper_index = 0
        
        for b in range(batch_size):
            
            upper_index = int(torch.sum(mask[:b]))
            temp_num_relation = int(mask[b])

            for k in self.relation_keys:
                out_gt[k].append( gt[k][b,:temp_num_relation,:] )

        for k in self.relation_keys:
            out_gt[k] = torch.cat(out_gt[k], 0)

        return predictions, out_gt


    # Accumulates all the tensors using mask in one matrix
    def make_tensor_random(self, predictions, gt, mask):

        '''
        This function calculates random performance
            pred: pred is a dictionary of predictions
            gt  : gt is dictionary of ground truth labels
            mask: mask is the places which should be considered 
        '''

        out_gt = {}
        out_pred = {}

        for k in self.relation_keys:
            out_pred[k] = []
            out_gt[k] = []

        num_relations = torch.split(mask, 1)

        for b, n in enumerate(num_relations):
            temp_num_relation = int(n)

            for k in self.relation_keys:
                out_gt[k].append( gt[k][b,:temp_num_relation,:] )
                temp_shape = predictions[k][b,:temp_num_relation,:].shape
                temp_device = predictions[k][b,:temp_num_relation,:].device
                if k != 'cr':
                    out_pred[k].append( torch.randint(2, temp_shape, device = temp_device) )
                if k == 'cr':
                    temp_vec = torch.randint(2, (temp_shape[0],), device = temp_device)
                    temp_vec_2 = F.one_hot(temp_vec, num_classes=3)
                    
                    out_pred[k].append( temp_vec_2 )

        for k in self.relation_keys:
            out_gt[k] = torch.cat(out_gt[k], 0)
            out_pred[k] = torch.cat(out_pred[k], 0)

        return out_pred, out_gt


    def calc_metrics(self, pred, gt):

        result = {}
        mask = gt['num_relation']
        
        if self.config['model_name'] == 'vsgnet':
            pred, gt = self.make_tensor_vsgnet(pred, gt, mask)
        else:
            pred, gt = self.make_tensor(pred, gt, mask)

        # Calculating mAP metrics
        result[self.mode + 'mAP_lr'] = self.metrics[self.mode + 'mAP_lr'](pred['lr'],gt['lr'])
        result[self.mode + 'mAP_mr'] = self.metrics[self.mode + 'mAP_mr'](pred['mr'],gt['mr'])
        result[self.mode + 'mAP_cr'] = self.metrics[self.mode + 'mAP_cr'](pred['cr'],gt['cr'])
        
        result[self.mode + 'mAP_all'] = result[self.mode + 'mAP_cr'] + result[self.mode + 'mAP_lr'] + result[self.mode + 'mAP_mr']
        result[self.mode + 'mAP_all']/=3

        # Calculating AP metrics
        AP_lr = self.metrics[self.mode + 'AP_lr'](pred['lr'],gt['lr'])
        for i, k in enumerate(self.lr_relations):
            result[self.mode + k + '_AP'] = AP_lr[i]


        AP_mr = self.metrics[self.mode + 'AP_mr'](pred['mr'],gt['mr'])
        for i, k in enumerate(self.mr_relations):
            result[self.mode + k + '_AP'] = AP_mr[i]

        AP_lr = self.metrics[self.mode + 'AP_lr'](pred['lr'],gt['lr'])
        
        for i, k in enumerate(self.lr_relations):
            result[self.mode + k + '_AP'] = AP_lr[i]

        return result
    
    def aggregate_metrics(self):
        result = {}
        
        # Calculating mAP metrics
        result[self.mode + 'mAP_lr'] = self.metrics[self.mode + 'mAP_lr'].compute()
        result[self.mode + 'mAP_mr'] = self.metrics[self.mode + 'mAP_mr'].compute()
        result[self.mode + 'mAP_cr'] = self.metrics[self.mode + 'mAP_cr'].compute()
        
        result[self.mode + 'mAP_all'] = result[self.mode + 'mAP_cr'] + result[self.mode + 'mAP_lr'] + result[self.mode + 'mAP_mr']
        result[self.mode + 'mAP_all']/=3

        # Calculating AP metrics
        AP_lr = self.metrics[self.mode + 'AP_lr'].compute()
        for i, k in enumerate(self.lr_relations):
            result[self.mode + k + '_AP'] = AP_lr[i]


        AP_mr = self.metrics[self.mode + 'AP_mr'].compute()
        for i, k in enumerate(self.mr_relations):
            result[self.mode + k + '_AP'] = AP_mr[i]


        AP_lr = self.metrics[self.mode + 'AP_lr'].compute()
        for i, k in enumerate(self.lr_relations):
            result[self.mode + k + '_AP'] = AP_lr[i]


        return result


    def reset_metrics(self):

        self.metrics[self.mode + 'mAP_lr'].reset()
        self.metrics[self.mode + 'mAP_mr'].reset()
        self.metrics[self.mode + 'mAP_cr'].reset()

        self.metrics[self.mode + 'AP_lr'].reset()
        self.metrics[self.mode + 'AP_mr'].reset()
        self.metrics[self.mode + 'AP_cr'].reset()

        self.last_result_cm = {}







class metric_tracker_non_ooi:

    def __init__(self, config, mode='train'):

        '''
        # Make Metric Objects
        # Needed metrics:

        average precision
        mAP
        confusion matrix 
        '''
        self.mode = mode
        self.config = config
        self.create_metrics()

    # Accumulates all the tensors using mask in one matrix
    def make_tensor(self, predictions, gt, mask):
        '''
        This function makes a combined tensor for ground truth and predictions using the mask

            pred: is a tensor of shape [batch, num_classes, time_steps]

            gt  : is a tensor of shape [batch, time_steps]

            mask: mask is a tensor of shape [batch]
        '''

        pred_return = []
        gt_return = []
        batch_size = mask.shape[0]

        for b in range(batch_size):

            num_frames = mask[b]

            temp_pred = predictions[b].transpose(0,1)[:num_frames, :]
            temp_gt = gt[b, :num_frames]
            
            pred_return.append(temp_pred)
            gt_return.append(temp_gt)

        pred_return = torch.concat(pred_return, dim=0)
        gt_return = torch.concat(gt_return, dim=0)
        
        
        return pred_return, gt_return

    def create_metrics(self):

        self.metrics = {}

        self.metrics[self.mode + '_frame_accuracy'] = tm.Accuracy(threshold=0.5,
                                                                   num_classes=self.config['num_classes'],
                                                                    average='macro').to(self.config['device'])

        self.metrics[self.mode + '_frame_mAP'] = tm.AveragePrecision(num_classes=self.config['num_classes'], 
                                                                         average='macro').to(self.config['device'])

    def calc_metrics(self, pred, gt, mask):
              
        pred, gt = self.make_tensor(pred, gt, mask)
        pred_classes = torch.argmax(pred, dim=1).tolist()
        gt_classes = gt.tolist()
        
        result = {}
        result[self.mode + '_frame_accuracy'] = self.metrics[self.mode + '_frame_accuracy'](pred, gt)
        result[self.mode + '_frame_mAP'] = self.metrics[self.mode + '_frame_mAP'](pred, gt)

        return result
    
    def aggregate_metrics(self):
        result = {}
        
        # Calculating mAP metrics
        result[self.mode + '_frame_accuracy'] = self.metrics[self.mode + '_frame_accuracy'].compute()
        result[self.mode + '_frame_mAP'] = self.metrics[self.mode + '_frame_mAP'].compute()
        return result

    def reset_metrics(self):
        
        for k in self.metrics.keys():
            self.metrics[k].reset()















class metric_tracker_multi_stream:

    def __init__(self, config, mode='train'):

        '''
        # Make Metric Objects
        # Needed metrics:

        average precision
        mAP
        confusion matrix 
        '''

        self.config = config
        device = self.config['device']

        self.relations = {}
        self.relations['cr'] = ['Contact', 'No Contact', 'None of these']

        self.relations['lr'] = ['Below/Above', 'Behind/Front', 'Left/Right', 'Inside', 'None of these']

        self.relations['mr'] = ['Holding', 'Carrying', 'Adjusting', 'Rubbing', 'Sliding', 'Rotating',
                       'Twisting', 'Raising', 'Lowering', 'Penetrating', 'Moving Toward', 
                       'Moving Away', 'Negligible Relative Motion', 'None of these']

        self.mode = mode + '_'

        self.metrics = {}
        
        self.relation_keys = ['mr', 'lr', 'cr']
        
        try:
            self.streams = self.config['streams']
        except:
            self.streams = ['']
        
        for stream in self.streams:
            for key in self.relation_keys:

                if stream != '':
                    stream_string = stream + '_'
                else:
                    stream_string = ''
                
                temp_mAP_key = self.mode + stream_string + 'mAP_' + key
                temp_AP_key = self.mode + stream_string + 'AP_' + key
                
                num_rel = len(self.relations[key])
                
                self.metrics[temp_mAP_key] = tm.AveragePrecision(num_rel, average='macro').to(device)
                self.metrics[temp_AP_key] = tm.AveragePrecision(num_rel, average=None).to(device)


    # Accumulates all the tensors using mask in one matrix
    def make_tensor(self, predictions, gt, mask):
        '''
        This function makes a combined tensor for ground truth and predictions using the mask

            pred: pred is a dictionary of predictions
            gt  : gt is dictionary of ground truth labels
            mask: mask is the places which should be considered 
        '''
        
        out_gt = {}

        for k in self.relation_keys:
            out_gt[k] = []

        # num_relations in different batch elements
        batch_size = mask.shape[0]

        
        for b in range(batch_size):

            temp_num_relation = int(mask[b])

            for k in self.relation_keys:
                out_gt[k].append( gt[k][b,:temp_num_relation,:] )

        for k in self.relation_keys:
            out_gt[k] = torch.cat(out_gt[k], 0)

        return predictions, out_gt




    # Accumulates all the tensors using mask in one matrix
    def make_tensor_gpnn(self, predictions, gt, mask):
        '''
        This function makes a combined tensor for ground truth and predictions using the mask

            pred: pred is a dictionary of predictions
            gt  : gt is dictionary of ground truth labels
            mask: mask is the places which should be considered 
        '''
        out_gt = {}
        out_pred = {}
        out_pred['combined'] = {}

        for k in self.relation_keys:
            out_pred['combined'][k] = []
            out_gt[k] = []

        # num_relations in different batch elements
        num_relations = torch.split(mask, 1)

        for b, n in enumerate(num_relations):

            temp_num_relation = int(torch.sum(n))
            # temp_num_relation = int(n)

            for k in self.relation_keys:
                out_gt[k].append( gt[k][b,:temp_num_relation,:] )
                out_pred['combined'][k].append( predictions['combined'][k][b,:temp_num_relation,:] )

        for k in self.relation_keys:
            out_gt[k] = torch.cat(out_gt[k], 0)
            out_pred['combined'][k] = torch.cat(out_pred['combined'][k], 0)

        return out_pred, out_gt


    def calc_metrics(self, pred, gt):

        result = {}
        mask = gt['num_relation']
        
        # if self.config['model_name'] == 'GPNN' or self.config['model_name'] == 'GPNN_icra' :
        #     pred, gt = self.make_tensor_gpnn(pred, gt, mask)
        # else:
        #     pred, gt = self.make_tensor(pred, gt, mask)

        pred, gt = self.make_tensor_gpnn(pred, gt, mask)

        
        # calculating mAP metrics
        
        for stream in self.streams:

            if stream != '':
                stream_string = stream + '_'
            else:
                stream_string = ''

            stream_all_mAP_result_key = self.mode + stream_string + 'mAP_all'
            result[ stream_all_mAP_result_key ] = 0

            for key in self.relation_keys:
                # Calculating mAP metrics
                temp_mAP_key = self.mode + stream_string + 'mAP_' + key
                result[temp_mAP_key] = self.metrics[temp_mAP_key]( pred[stream][key], gt[key] )
                result[ stream_all_mAP_result_key ] += result[temp_mAP_key]
                
                # Calculating AP metrics
                temp_AP_key = self.mode + stream_string + 'AP_' + key

                temp_AP = self.metrics[temp_AP_key](pred[stream][key], gt[key])
                relations_list = self.relations[key]
                
                for i, rel in enumerate(relations_list):
                    rel_key = self.mode + stream_string + 'AP_' + rel
                    result[rel_key] = temp_AP[i]

            result[ stream_all_mAP_result_key ]/=3.0

        return result
    
    def aggregate_metrics(self):

        result = {}

        for stream in self.streams:

            if stream != '':
                stream_string = stream + '_'
            else:
                stream_string = ''

            stream_all_mAP_result_key = self.mode + stream_string + 'mAP_all'
            result[ stream_all_mAP_result_key ] = 0

            for key in self.relation_keys:

                # Calculating mAP metrics
                temp_mAP_key = self.mode + stream_string + 'mAP_' + key
                result[temp_mAP_key] = self.metrics[temp_mAP_key].compute()
                result[ stream_all_mAP_result_key ] += result[temp_mAP_key]
                
                # Calculating AP metrics
                temp_AP_key = self.mode + stream_string + 'AP_' + key

                temp_AP = self.metrics[temp_AP_key].compute()
                relations_list = self.relations[key]
                
                for i, rel in enumerate(relations_list):
                    rel_key = self.mode + stream_string + 'AP_' + rel
                    result[rel_key] = temp_AP[i]

            result[ stream_all_mAP_result_key ]/=3.0

        return result


    def reset_metrics(self):

        for stream in self.streams:

            if stream != '':
                stream_string = stream + '_'
            else:
                stream_string = ''

            for key in self.relation_keys:

                # Calculating mAP metrics
                temp_mAP_key = self.mode + stream_string + 'mAP_' + key
                self.metrics[temp_mAP_key].reset()
                
                # Calculating AP metrics
                temp_AP_key = self.mode + stream_string + 'AP_' + key

                self.metrics[temp_AP_key].reset()

        self.last_result_cm = {}