import torch
from legodnn.utils.common.file import ensure_dir

config_file_name = 'deeppose/mpii/res50_mpii_256x256.py'

if __name__=='__main__':
    cv_task = 'pose_estimation'
    device = 'cuda'
    root_path = 'cv_task/pose_estimation/mmpose_models/configs/body/2d_kpt_sview_rgb_img/'
    config_file = root_path + config_file_name
    model_name = config_file_name.split('/')[0]
    checkpoint = None
    
    from mmpose.apis import init_pose_model
    model = init_pose_model(config_file, checkpoint, device=device)
    print(model)
    model_save_path = '/'.join(['./results', cv_task, model_name, 'model.pt'])
    ensure_dir(model_save_path)
    torch.save(model, model_save_path)