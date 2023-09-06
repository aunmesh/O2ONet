import wandb
import os

class wandb_logger:

    def __init__(self, config):

        self.wandb_log = 1

        proj_name = config['wandb_project_name']
        entity = config['wandb_entity']

        if 'run_id' not in config.keys():
            run_id = wandb.util.generate_id()
            wandb.init(project=proj_name, entity=entity, id=run_id, resume="allow", config=config)
            wandb.config.update({"run_id": run_id})
            config['run_id'] = run_id
            self.config = config

        else:
            run_id = config['run_id']
            wandb.init(project=proj_name, entity=entity, id=run_id, resume="allow", config=config)
            
            try:
                wandb.restore( os.path.join(config['model_saving_path'], run_id, 'checkpoint.pth') )
            except:
                pass
            
            self.config = config

    
    def watch(self, model):
        wandb.watch(model, log_freq=1)

    def log_dict(self, log_dict):

        wandb.log(log_dict)


'''
config = config_loader(config_file)
wandb_log = 1
if wandb_log ==0:
    run_id = 'qwerty'
if wandb_log:
    if len(sys.argv) > 2:
        run_id = sys.argv[2]
    else:
        run_id = wandb.util.generate_id()
    print("RUN id is" + str(run_id))
    model_saving_path = os.path.join(config['model_saving_path'],run_id)
    os.makedirs(model_saving_path, exist_ok=True)
    wandb.init(project="eccv_22_gcn", entity="cdeslab_ooi_interaction", id=run_id, resume="allow", config=config)
wandb.config.update({"run_id": run_id})

if wandb_log:
    wandb.watch(ooi_net_1, log_freq=1)



    if wandb_log:
        
        loss_dict_train = {"train_loss": loss['all'], "train_loss_scr": loss['scr'],
                        "train_loss_mr": loss['mr'], "train_loss_lr": loss['lr']}
        
        
        loss_dict_val = {"val_loss": val_loss['all'], "val_loss_scr": val_loss['scr'], 
                   "val_loss_mr": val_loss['mr'], "val_loss_lr": val_loss['lr']}

        val_lr_tables = make_lr_tables(val_metrics_cm['val_cm_lr'], 'val')
        val_mr_tables = make_mr_tables(val_metrics_cm['val_cm_mr'], 'val')
        val_scr_tables = make_scr_tables(val_metrics_cm['val_cm_scr'], 'val')

        train_lr_tables = make_lr_tables(train_metrics_cm['train_cm_lr'], 'train')
        train_mr_tables = make_mr_tables(train_metrics_cm['train_cm_mr'], 'train')
        train_scr_tables = make_scr_tables(train_metrics_cm['train_cm_scr'], 'train')

        log_dict = {**loss_dict_train, **train_metrics, **loss_dict_val, **val_metrics, **train_lr_tables
                    ,**train_mr_tables, **train_scr_tables, **val_lr_tables, **val_mr_tables, 
                    **val_scr_tables}

        wandb.log(log_dict)

'''