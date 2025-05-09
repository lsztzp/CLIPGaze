from timeit import default_timer as timer
import argparse
from datetime import datetime
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from models import Transformer
from CLIPGaze import CLIPGaze
from utils import seed_everything, fixations2seq, get_args_parser_train, save_model_train, cutFixOnTarget
from dataset import fixation_dataset, COCOSearch18Collator
torch.autograd.set_detect_anomaly(True)
from test import run_model
from metrics import postprocessScanpaths, get_seq_score, get_ed, get_semantic_ed, get_semantic_seq_score,compute_mm, compute_spatial_metrics_by_step, get_seq_score_time,get_ed_time,get_semantic_seq_score_time,get_semantic_ed_time
from torch.utils.tensorboard import SummaryWriter
import pickle



def train(epoch, args, model, SlowOpt, MidOpt, FastOpt, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, train_dataloader, model_dir, model_name, device = 'cuda:0', im_h=20, im_w=32, project_num=16):
    model.train()
    token_losses = 0
    reg_losses = 0
    t_losses = 0


    with tqdm(train_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix in tepoch:
            out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks)
            out_y, out_x = torch.clamp(out_y, min=0, max=im_h * project_num - 2), torch.clamp(out_x, min=0, max=im_w * project_num - 2)

            SlowOpt.zero_grad()
            MidOpt.zero_grad()
            FastOpt.zero_grad()

            tgt_out = batch_tgt.to(device)
            batch_tgt_padding_mask = batch_tgt_padding_mask.to(device)
            token_gt = batch_tgt_padding_mask.long()
            fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            #predict padding or valid fixation
            token_loss = loss_fn_token(out_token.permute(1, 2, 0), token_gt)
            # token_loss = loss_fn_token(out_token.squeeze(-1).permute(1,0), token_gt.float())

            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask

            out_yx = torch.stack((out_y, out_x), dim=2)
            tgt_out_yx = tgt_out[:, :,:-1] * torch.stack((fixation_mask,fixation_mask),dim=2)
            lens_arr = torch.argmax(token_gt, dim=1)
            lens_arr = torch.where(lens_arr == 0, torch.tensor(7).to(device), lens_arr)
            batch_size, _, _ = out_yx.shape
            dtw_loss = 0
            for b in range(batch_size):
                dtw_distances = torch.cdist(tgt_out_yx[b][:lens_arr[b],:], out_yx[b][:lens_arr[b],:])
                dtw_path = dtw_distances.min(dim=1)[0].mean()  # 计算最小距离并取平均
                dtw_loss += dtw_path
            dtw_loss /= batch_size
            # beta=0.5

            #calculate regression L1 losses for only valid ground truth fixations
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()

            # loss = beta * dtw_loss + (1-beta)*reg_loss + token_loss + t_loss
            loss = dtw_loss + reg_loss + t_loss + token_loss

            loss.backward()
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()

            SlowOpt.step()
            MidOpt.step()
            FastOpt.step()
            
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch)
    if epoch>=60 and epoch % 10==0:
        save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name)
    return token_losses / len(train_dataloader),  reg_losses / len(train_dataloader), t_losses / len(train_dataloader)

def evaluate(model, device = 'cuda:0', im_h=20, im_w=32, project_num=16):
    model.eval()

    fixation_path = "./dataset/coco_search18_fixations_TP_test.json"
    if args.condition == 'absent':
        fixation_path = "./dataset/coco_search18_fixations_TA_test.json"

    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)

    if args.condition == 'present':
        test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition'] == "present",human_scanpaths))  # 1
    else:
        test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition'] == "absent", human_scanpaths))  # 1

    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    test_task_img_pairs = np.unique(
        [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])

    embedding_dict = np.load(open(args.embedding_dir, mode='rb'), allow_pickle=True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(1))


    if args.condition == "present":
        img_ftrs_dir = args.img_ftrs_dir
    else:
        img_ftrs_dir = args.img_ftrs_dir_absent

    for target_traj in tqdm(test_task_img_pairs):
        task_name, name, condition = target_traj.split('_')

        image_ftrs = torch.load(join(img_ftrs_dir, task_name.replace(' ', '_'), name.replace('jpg', 'pth')))
        image_ftrs = [image_ftrs]
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, im_h=im_h, im_w=im_w,
                              project_num=project_num, num_samples=1)
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx + 1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('./data', 'clusters.npy'), allow_pickle=True).item()

    max_len = args.max_len
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)

    FED = get_ed(predictions, fix_clusters, max_len)

    if args.condition == 'present':
        with open('./segmentation_map_dir/SemSS/test_TP_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    elif args.condition == 'absent':
        with open('./segmentation_map_dir/SemSS/test_TA_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    print("Calculating SemSS,SemFED,mm...")
    SemSS = get_semantic_seq_score(predictions, fixations_dict, max_len, './segmentation_map_dir/SemSS/stuffthing_maps')
    SemFED = get_semantic_ed(predictions, fixations_dict, max_len, './segmentation_map_dir/SemSS/stuffthing_maps')

    if args.condition == 'absent':
        for x in test_target_trajs:
            x['X']=[a/1680*512 for a in x['X']]
            x['Y']=[a/1050*320 for a in x['Y']]

    mm = compute_mm(test_target_trajs, predictions, 512, 320)
    mm = mm[:-1].mean()
    print("Calculating cc,nss...")
    cc, nss = compute_spatial_metrics_by_step(predictions, test_target_trajs)

    return seq_score, FED, SemSS, SemFED, mm, cc, nss
    
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.cuda))
    device_id = args.cuda
    retraining = args.retraining

    assert args.work_dir
    if retraining:
        args.work_dir = args.reload_path
        args.cuda = device_id
    else:
        args.work_dir = os.path.join('/02-Results/01-ScanPath/logs/gazeformer/',
                                     datetime.today().strftime('%m-%d-') + args.work_dir)
    writer = SummaryWriter(log_dir=args.work_dir)
    logfile = join(args.work_dir, 'output.txt')
    scorefile = join(args.work_dir, 'score.txt')

    model_dir = join(args.work_dir, "checkpoints")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print(str(vars(args)) + '\n\n')
    with open(join(model_dir, 'config.json'), "w") as outfile:
        json.dump(vars(args), outfile)
        outfile.close()

    model_name = 'gazeformer_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'_'+str(args.hidden_dim)+'d'
    dataset_root = args.dataset_dir
    if args.condition == "present":
        train_file = args.train_file
    else:
        train_file = args.train_file_absent
    with open(join(dataset_root, train_file)) as json_file:
        fixations_train = json.load(json_file)

    fixations_train = fixations2seq(fixations=fixations_train, max_len=args.max_len)
    for traj in fixations_train:
        traj['tgt_seq_x'][0] = args.im_w//2*args.project_num  #256
        traj['tgt_seq_y'][0] = args.im_h//2*args.project_num  #160
    if args.condition=="present":  #has 2661 not find the target,wrong data
        bbox_annos = np.load(args.bbox_file, allow_pickle=True).item()
        seq_train = cutFixOnTarget(fixations_train, bbox_annos)
    else:
        seq_train = fixations_train

    ratio = sum(len(x['tgt_seq_x']) for x in seq_train) / len(seq_train) - 1
    print("termination pos weight: {:.3f}".format(ratio))

    if args.condition == "present":
        train_dataset = fixation_dataset(seq_train, img_ftrs_dir = args.img_ftrs_dir)
    else:
        train_dataset = fixation_dataset(seq_train, img_ftrs_dir = args.img_ftrs_dir_absent)

    # target embeddings
    embedding_dict = np.load(open(args.embedding_dir, mode='rb'), allow_pickle=True).item()
    collate_fn = COCOSearch18Collator(embedding_dict,args.max_len, args.im_h, args.im_w, args.project_num)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=6, collate_fn = collate_fn)

    transformer = Transformer(nhead = args.nhead, d_model = args.hidden_dim,
    num_decoder_layers=args.num_decoder, decoder_dropout = args.decoder_dropout, dim_feedforward = args.hidden_dim,
    device = device).to(device)

    model = CLIPGaze(transformer, spatial_dim = (args.im_h, args.im_w), dropout=args.cls_dropout, max_len = args.max_len, device = device).to(device)

    weights = torch.tensor([1.0, ratio]).to(device)
    loss_fn_token = torch.nn.NLLLoss(weight=weights)
    loss_fn_y = nn.L1Loss(reduction='none')
    loss_fn_x = nn.L1Loss(reduction='none')
    loss_fn_t = nn.L1Loss(reduction='none')

    #Disjoint optimization
    head_params = list(model.token_predictor.parameters()) + list(model.transformer.film_add.parameters()) + list(model.transformer.film_mul.parameters()) + list(model.transformer.reduces.parameters()) + list(model.transformer.blocks.parameters()) + list(model.transformer.extra_blocks.parameters())
    SlowOpt = torch.optim.AdamW( head_params, lr=args.head_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    belly_params = list(model.generator_t_mu.parameters())
    MidOpt = torch.optim.AdamW(belly_params, lr=args.belly_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    tail_params = list(model.transformer.decoder.parameters())  + list(model.generator_y_mu.parameters()) + list(model.generator_x_mu.parameters()) + list(model.querypos_embed.parameters()) + list(model.firstfix_linear.parameters())
    FastOpt = torch.optim.AdamW(tail_params, lr=args.tail_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    start_epoch = 1
    best_epoch, best_score = 0, 0
    if retraining:
        checkpoint = torch.load(args.reload_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        SlowOpt.load_state_dict(checkpoint['optim_slow'])
        MidOpt.load_state_dict(checkpoint['optim_mid'])
        FastOpt.load_state_dict(checkpoint['optim_fast'])

        SlowOpt.param_groups[0]['capturable'] = True
        MidOpt.param_groups[0]['capturable'] = True
        FastOpt.param_groups[0]['capturable'] = True

        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)
    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        train_token_loss, train_reg_loss, train_t_loss = train(epoch = epoch, args = args, model = model, SlowOpt = SlowOpt, FastOpt = FastOpt, MidOpt = MidOpt, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t = loss_fn_t, train_dataloader = train_dataloader, model_dir = model_dir, model_name = model_name, device = device)
        end_time = timer()

        ####################################
        writer.add_scalar("AA_Scalar/train_token_loss", train_token_loss, epoch)
        writer.add_scalar("AA_Scalar/train_reg_loss", train_reg_loss, epoch)
        writer.add_scalar("AA_Scalar/train_t_loss", train_t_loss, epoch)
        ####################################

        seq_score, FED, SemSS, SemFED, mm, cc, nss = evaluate(model = model, device = device)
        ######################################
        writer.add_scalar("AA_Scalar/SS", seq_score, epoch)
        writer.add_scalar("AA_Scalar/FED", FED, epoch)
        writer.add_scalar("AA_Scalar/SemSS", SemSS, epoch)
        writer.add_scalar("AA_Scalar/SemFED", SemFED, epoch)
        writer.add_scalar("AA_Scalar/mm", mm, epoch)
        writer.add_scalar("AA_Scalar/cc", cc, epoch)
        writer.add_scalar("AA_Scalar/nss", nss, epoch)
        #######################################

        output_str = f"Epoch: {epoch}, Train token loss: {train_token_loss:.3f}, Train reg loss: {train_reg_loss:.3f}, Train T loss: {train_t_loss:.3f}, SS: {seq_score:.3f},  SemSS: {SemSS:.3f}, SemFed: {SemFED:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"
        print(output_str)
        with open(logfile, "a") as myfile:
            myfile.write(output_str)
            myfile.close()

        if SemSS >= best_score:
            best_epoch=epoch
            best_score=SemSS
            save_path = os.path.dirname(model_dir.rstrip('/'))+"/best_performance.pkg"
            save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name,save_path=save_path)
            score_str = f"Best_Epoch_is: {best_epoch}, SS: {seq_score:.3f}, FED: {FED:.3f}, SemSS: {SemSS:.3f}, SemFED: {SemFED:.3f},  mm: {mm:.3f}, CC: {cc:.3f}, NSS: {nss:.3f}\n"
            with open(scorefile, "a") as myfile:
                myfile.write(score_str)
                myfile.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIPGaze Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)
    
