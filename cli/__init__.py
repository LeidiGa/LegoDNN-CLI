import click
import torch
import sys
sys.setrecursionlimit(100000)
from legodnn import AutoBlockManager, CommonDetectionManager, CommonModelManager, BlockExtractor, topology_extraction
from legodnn.utils.common.file import ensure_dir
    
@click.group()
def legodnn():
    pass
        
@click.command()
@click.option('--path', '-p', prompt='Model path', type=click.Path(exists=True), 
              help='Path to the original model.')
@click.option('--ratio', '-r', default=0.125, 
              help='Maximum ratio of compressed layer.')
@click.option('--shape', '-s', prompt='Model input shape', type=click.STRING, 
              help='String of model input shape, three-dimensional for image and four-dimensional for video. Numbers should be separated by \',\' such as 3,32,32')
@click.option('--output', '-o', type=click.Path(exists=False), 
              help='Path to save blocks.')
@click.option('--device', '-d', default='cuda', type=click.STRING, 
              help='Device to save blocks.')
def extract(path, ratio, shape, output, device):
    # blocks save path
    if output == None:
        output = '/'.join(path.split('/')[:-1]) + '/blocks'
    ensure_dir(output)
    
    # input shape
    model_input_shape = (1, ) + tuple([int(s) for s in shape.split(',')])
    model = torch.load(path)
    
    # handle mmcv model
    if hasattr(model, 'forward_dummy'):
        click.echo('MMCV Model: Using forward method \'forward_dummy\'')
        model.forward = model.forward_dummy
    elif hasattr(model, 'forward_test'):
        click.echo('MMCV Model: Using forward method \'forward_test\'')
        model.forward = model.forward_test
    
    # build graph
    model_graph = topology_extraction(model, model_input_shape, device=device, mode='unpack')
    
    # block detection
    detection_manager = CommonDetectionManager(model_graph, max_ratio=ratio)
    detection_manager.detection_all_blocks()
    
    # block extration
    model_manager = CommonModelManager()
    block_manager = AutoBlockManager([0], detection_manager, model_manager)
    block_extractor = BlockExtractor(model, block_manager, output, shape, device)
    block_extractor.extract_all_blocks()
    
    # show blocks info
    # for i, block_id in enumerate(block_manager.get_blocks_id()):
    #     for block_sparsity in block_manager.get_blocks_sparsity()[i]:
    #         block_file_name = block_manager.get_block_file_name(block_id, block_sparsity)
    #         block_file_path = os.path.join(output, block_file_name)
    #         print(block_manager.get_block_from_file(block_file_path, device), block_file_name)


legodnn.add_command(extract)

if __name__ == '__main__':
    legodnn()