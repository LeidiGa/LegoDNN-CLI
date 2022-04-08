import torch
from legodnn.utils.common.file import ensure_dir

config_file_name = 'fcn/fcn_r50-d8_512x1024_40k_cityscapes.py'

if __name__=='__main__':
    cv_task = 'semantic_segmentation'
    device = 'cuda'
    root_path = 'cv_task/semantic_segmentation/mmseg_models/configs/'
    config_file = root_path + config_file_name
    model_name = config_file_name.split('/')[0]
    checkpoint = None
    
    from mmseg.apis import init_segmentor
    model = init_segmentor(config_file, checkpoint, device=device)
    print(model)
    model_save_path = '/'.join(['./results', cv_task, model_name, 'model.pt'])
    ensure_dir(model_save_path)
    torch.save(model, model_save_path)