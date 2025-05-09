import argparse
import os
import random
from os.path import join
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fixations2seq(fixations, max_len):
    processed_fixs = []
    num = 0
    for fix in fixations:
        if len(fix['X']) <= max_len:
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[:max_len]),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[:max_len]),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[:max_len]),
                                   'task': fix['task'], 'img_name': fix['name']})
        else:
            num += 1
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[-max_len:]),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[-max_len:]),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[-max_len:]),
                                   'task': fix['task'], 'img_name': fix['name']})
    print("Has:%d scanpath over length" % num)
    return processed_fixs

def get_args_parser_test():
    parser = argparse.ArgumentParser('Gaze Transformer Tester', add_help=False)
    parser.add_argument('--dataset_dir', default='./dataset', type=str, help="Dataset Directory")
    parser.add_argument('--img_ftrs_dir', default='/01-Datasets/01-ScanPath-Datasets/coco_search18/vit-L14-336/featuremaps/',
                        type=str, help="Directory of precomputed target present CLIP visual features")
    parser.add_argument('--img_ftrs_dir_absent', default='/01-Datasets/01-ScanPath-Datasets/coco_search18/vit-L14-336/featuremaps_TA/',
                        type=str, help="Directory of precomputed target absent CLIP visual features")
    parser.add_argument("--embedding_dir", default="./dataset/embeddings.npy",
                        type=str, help="embedding_dir")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--project_num', default=16, type=int,
                        help="The num to project to the feature map with image dimensions (320X512)")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=1024, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--trained_model', default='./checkpoint/CLIPGaze_TP.pkg', type=str,
                        help="Trained model checkpoint to run for inference")
    parser.add_argument('--seed', default=3407, type=int, help="Seed")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--condition', default='present', type=str, help="Search condition (present/absent)")
    parser.add_argument('--zerogaze', default=False, action='store_true', help="ZeroGaze setting flag")
    parser.add_argument('--task', default='car', type=str, help="if evaluation is in ZeroGaze setting, the unseen target to evaluate the model")
    parser.add_argument('--num_samples', default=1, type=int, help="Number of scanpaths sampled per test case")

    return parser
    

def get_args_parser_train():
    parser = argparse.ArgumentParser('Gaze Transformer Trainer', add_help=False)
    parser.add_argument('--head_lr', default=1e-6, type=float, help="Learning rate for SlowOpt")
    parser.add_argument('--tail_lr', default=1e-4, type=float, help="Learning rate for FastOpt")
    parser.add_argument('--belly_lr', default=2e-6, type=float, help="Learning rate for MidOpt")
    parser.add_argument('--dataset_dir', default='./dataset', type=str, help="Dataset Directory")

    parser.add_argument('--condition', default='present', type=str, help="Search condition (present/absent)")  # 1
    parser.add_argument('--train_file', default='coco_search18_fixations_TP_train.json', type=str, help="Training fixation file")
    parser.add_argument('--train_file_absent', default='coco_search18_fixations_TA_train.json', type=str, help="Training fixation file")
    parser.add_argument('--bbox_file', default='./dataset/bbox_annos.npy', type=str, help="Training fixation file")
    parser.add_argument('--img_ftrs_dir', default='/01-Datasets/01-ScanPath-Datasets/coco_search18/vit-L14-336/featuremaps/',
                        type=str, help="Directory of precomputed target present CLIP features")
    parser.add_argument('--img_ftrs_dir_absent', default='/01-Datasets/01-ScanPath-Datasets/coco_search18/vit-L14-336/featuremaps_TA/',
                        type=str, help="Directory of precomputed target absent CLIP features")
    parser.add_argument("--embedding_dir", default="./dataset/embeddings.npy",
                        type=str, help="embedding_dir")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--project_num', default=16, type=int,
                        help="The num to project to the feature map with image dimensions (320X512)")
    parser.add_argument('--seed', default=3407, type=int, help="seed")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch Size")
    parser.add_argument('--epochs', default=250, type=int, help="Maximum number of epochs to train")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=1024, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
    parser.add_argument('--cls_dropout', default=0.2, type=float, help="Final scanpath prediction dropout rate")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--num_workers', default=6, type=int, help="Number of workers for data loader")
    parser.add_argument("--work_dir", default="baseline", type=str, help="log_dir")
    parser.add_argument('--retraining', default=False, action='store_true', help="Retraining from a checkpoint")
    parser.add_argument("--reload_path", default="./checkpoint/best_performance_tp.pkg", type=str, help="log_dir")

    return parser

def save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name, save_path="",
                     only_trainable=True):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
            model.module.state_dict()
            if hasattr(model, "module") else model.state_dict(),
        "optim_slow":
            SlowOpt.state_dict(),
        "optim_mid":
            MidOpt.state_dict(),
        "optim_fast":
            FastOpt.state_dict(),
    }
    if only_trainable:
        weight_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        name = model.named_parameters()
        state = {
            "epoch": epoch,
            "args": vars(args),
            "model": {n: weight_dict[n] for n, p in name if p.requires_grad},
            "optim_slow":
                SlowOpt.state_dict(),
            "optim_mid":
                MidOpt.state_dict(),
            "optim_fast":
                FastOpt.state_dict(),
        }
    if save_path:
        torch.save(state, save_path)
    else:
        torch.save(state, join(model_dir, model_name + '_' + str(epoch) + '.pkg'))

def cutFixOnTarget(trajs, target_annos):
    processed_trajs = []
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    if len(trajs) == 0:
        return
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['img_name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['tgt_seq_x'], traj['tgt_seq_y'], bbox)
            num_steps_task[i] = traj_len
            traj['tgt_seq_x'] = traj['tgt_seq_x'][:traj_len]
            traj['tgt_seq_y'] = traj['tgt_seq_y'][:traj_len]
            traj['tgt_seq_t'] = traj['tgt_seq_t'][:traj_len]
            if traj_len != 100:
                processed_trajs.append(traj)
    print('data cuted')
    return processed_trajs

def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 100
