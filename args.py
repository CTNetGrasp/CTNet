import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')


    # Dataset & Data 
    parser.add_argument('--img_size', type=int, default=320, help='image size')

    parser.add_argument('--use_depth', action="store_true", help='Use Depth image for training (1/0)')
    parser.add_argument('--use_rgb', action="store_false", help='Use RGB image for training (0/1)')
    parser.add_argument('--use_box',  action="store_true", help='Use box annotation for training (0/1)')
    parser.add_argument('--use_mask', action="store_true", help='Use mask annotation for training (0/1)')
    
    parser.add_argument('--random_rotate', action="store_true", help='Use rotate enhancement for training (0/1)')
    parser.add_argument('--random_crop', action="store_true", help='Use crop enhancement for training (0/1)')
    parser.add_argument('--random_bright', action="store_true", help='Use bright enhancement for training (0/1)')

    parser.add_argument('--txt_length', type=int, default=15, help='the length of words')
    parser.add_argument('--tokenizer', type=str, default='clip', help='tokenizer of text preprocess: clip or bert')

    parser.add_argument('--workers', type=int, default=16, help='thread of dataloader')
    parser.add_argument('--split', type=str, default='val', help='split of datasets')



    #attribute of network 
    parser.add_argument('--clip_pretrain', type=str, default='pretrain/CLIP/RN50.pt', help='pretraining weight of CLIP')
    parser.add_argument('--word_dim', type=int, default=512, help='channel of word embedding')
    parser.add_argument('--sent_dim', type=int, default=1024, help='channel of sentence vector')
    parser.add_argument('--feats_dims', type=list, default=[2048, 1024, 512, 256], help='output channel of hierarchical visual encoder')
    
    parser.add_argument('--img_dim', type=int, default=1024, help='channel of visual vector in encoder')
    parser.add_argument('--seg_dim', type=int, default=64, help='channel of segmentation branch')
    parser.add_argument('--grasp_dim', type=int, default=64, help='channel of grasp branch')


    #hardware
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--device', type=int, default='0', help='specify the gpu device for one gpu training')
    
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip_lr', type=float, default=1e-4, help='learning rate for clip parameters')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--milestones', type=list, default=[8, 10, 12], help='execution time of learing schedule')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay of learing schedule')
    parser.add_argument('--resume', type=str, default="", help='path of checkpoint for resume or inference')


    # save & visualization
    parser.add_argument('--print_freq', type=int, default=100, help='print frequence')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Training Output Directory')
    parser.add_argument('--visualize', action="store_true", help='viualize the grasp, mask of prediction and ground truth')
    parser.add_argument('--model_name', type=str, default='CGnet', help='name of checkpoint')

    parser.add_argument('--log_name', type=str, default='', help='name of checkpoint')



    args = parser.parse_args()
    return args
