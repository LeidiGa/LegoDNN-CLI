import torch
from legodnn.utils.common.file import ensure_dir

config_file_name = 'resnet/resnet50_8xb16_cifar10.py'

if __name__=='__main__':
    cv_task = 'image_classification'
    device = 'cuda'
    root_path = 'cv_task/image_classification/mmcls_models/configs/'
    config_file = root_path + config_file_name
    model_name = config_file_name.split('/')[0]
    checkpoint = None
    
    from mmcls.apis import init_model
    model = init_model(config_file, checkpoint, device=device)
    print(model)
    model_save_path = '/'.join(['./results', cv_task, model_name, 'model.pt'])
    ensure_dir(model_save_path)
    torch.save(model, model_save_path)