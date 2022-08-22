from model.nn_nets.ooi_net import ooi_net as ooi_net
import torch
import os


def get_model(config):

    if config['model_name'] == 'ooi_net':

        model = ooi_net(config).to(config['device'])
        return model


from dataloader.dataset import dataset


def get_dataset(config, split='train'):

    if config['dataset_description'] == 'ooi_dataset':

        return dataset(config, split)



def freeze_layers(model, freezing_list):

    named_params = [p for p in model.named_parameters()]

    for temp_param in named_params:
        if temp_param[0] in freezing_list:
            temp_param[1].requires_grad=False

def unfreeze_model(model):

    named_params = [p for p in model.named_parameters()]

    for temp_param in named_params:
        temp_param[1].requires_grad=True
    

def get_optimizer(config, model):

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return optimizer

def load_checkpoint(config, model, optimizer):

    checkpoint_path = os.path.join(config['model_saving_path'], config['run_id'], 'checkpoint.pth')

    try:
        f_obj = open(checkpoint_path,'rb')
        checkpoint = torch.load(f_obj)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer, start_epoch
    
    except:
        return model, optimizer, 0


def save_state(model, optimizer, epoch, config, best=False):

    os.makedirs(os.path.join(config['model_saving_path'], config['run_id']), exist_ok=True)
    checkpoint_path = os.path.join(config['model_saving_path'], config['run_id'], 'checkpoint.pth')
    best_model_path = os.path.join(config['model_saving_path'], config['run_id'], 'best_model.pth')

    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)

    if best:
        torch.save(checkpoint, best_model_path)