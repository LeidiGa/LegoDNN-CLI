import torch
from legodnn.utils.common.file import ensure_dir

config_file_name = 'tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb.py'

if __name__=='__main__':
    cv_task = 'action_recognition'
    device = 'cuda'
    root_path = 'cv_task/action_recognition/mmaction_models/configs/recognition/'
    config_file = root_path + config_file_name
    model_name = config_file_name.split('/')[0]
    checkpoint = None
    
    from mmaction.apis import init_recognizer
    model = init_recognizer(config_file, checkpoint, device=device)
    print(model)
    model_save_path = '/'.join(['./results', cv_task, model_name, 'model.pt'])
    ensure_dir(model_save_path)
    torch.save(model, model_save_path)