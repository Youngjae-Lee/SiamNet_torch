import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import visdom
from imageio import imread
from tqdm import tqdm

from utils.image_utils import *
from utils.utils import *
from model import *
from dataloader import ImageNetVID, ImageNetVID_val
from losses import *
from labels import *
from metrics import *
from optimizer import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', required=True, type=str,
                        help="ImageNetVID root directory")
    parser.add_argument('-p', '--param-path', required=True, type=str,
                        help="Parameter Json File")
    parser.add_argument('--port', required=False, type=int, default=8097,
                        help="Visdom Port(default:8097)")
    parser.add_argument('-t', '--pre-trained', required=False, type=str, default="",
                        help="Continue to training")
    parser.add_argument('-g', '--gpus', required=False, default='0', type=str,
                        help="GPU Number")
    args = parser.parse_args()
    return args


def display_images(ref, srch, score_map, label,type='train', viz=None):
    if viz is None:
        return
    opts = {"colormap": 'Magma'}
    viz.image(ref, win="{0}_ref".format(type), opts=dict(title="Reference"))
    viz.image(srch, win='{0}_srch'.format(type), opts=dict(title="Search"))
    viz.heatmap(score_map[0], 'heatmap', opts=opts)
    viz.heatmap(label[:, :, 0], 'pos_label', opts=opts)
    viz.heatmap(label[:, :, 1], 'neg_label', opts=opts)


def plot_2d_line(vals, iter, epoch, total_batch_per_epoch, type='train', kinds='loss', viz=None):
    x_axis = epoch*total_batch_per_epoch + iter
    win_name = "{0}_{1}".format(type, kinds)
    viz.line(X=np.array([x_axis]), Y=np.array([vals]), win=win_name, update='append', opts=dict(title=win_name))


def train_and_evaluate(model, train_loader, eval_loader, optim, loss_func, sched, metrics, **kwargs):
    start = kwargs['start_epoch']
    end = kwargs['total_epoch']
    device = kwargs['device']
    params = kwargs['param']
    if 'viz' in kwargs:
        viz = kwargs['viz']
    else:
        viz = None

    save_data = dict(model=model.state_dict(), optim=optim.state_dict(), scheduler=sched.state_dict(), epoch=0)
    for epoch in tqdm(range(start, end), initial=start):
        train(model, train_loader, optim, loss_func, metrics, epoch=epoch, device=device, viz=viz, display_step=params.display_step)
        sched.step()
        save_data.update(
            {"model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch}
        )
        save_model(path_to_save=params.ckpt_path, save_params=save_data)
        if eval_loader is not None:
            evaluate(model, eval_loader, loss_func, metrics, device, epoch=epoch, viz=viz)

    print("End Training")


def train(model, loader, optim, loss_func, metrics, device, **kwargs):
    model.train()
    avg = RunningAverageMultiVar(loss=RunningAverage(), auc=RunningAverage())
    progbar = tqdm(loader)
    disp_step = kwargs['display_step']
    viz = kwargs['viz']
    epoch = kwargs['epoch']
    for idx, sample in enumerate(progbar):
        ref_img_batch = sample['ref'].to(device)
        srch_img_batch = sample['srch'].to(device)
        label_batch = sample['label'].to(device)
        score_map = model(ref_img_batch, srch_img_batch)
        loss = loss_func(score_map=score_map, labels=label_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_val = loss.to('cpu').item()
        if idx % disp_step == 0 and viz is not None:
            display_images(sample['ref'][0], sample['srch'][0], score_map.to('cpu')[0], sample['label'][0], viz=viz)
        auc_val = metrics(score_map.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
        avg.update(loss=loss_val, auc=auc_val)
        plot_2d_line(loss_val, idx, epoch, len(loader),type='train', kinds='loss', viz=viz)
        plot_2d_line(auc_val, idx, epoch, len(loader),type='train', kinds='auc', viz=viz)
        progbar.set_postfix({"Loss": "%0.4f"%avg['loss'](), "AUC Score": "%0.3f"%avg['auc']()})
        torch.cuda.empty_cache()

    return avg['loss'](), avg['auc']()

@torch.no_grad()
def evaluate(model, loader, loss_func, metrics, device, **kwargs):
    epoch = kwargs['epoch']
    model.eval()
    avg = RunningAverageMultiVar(loss=RunningAverage(), auc=RunningAverage())
    viz = kwargs['viz']
    progbar = tqdm(loader)
    for idx, sample in enumerate(progbar):
        ref_image = sample['ref'].to(device)
        srch_image = sample['srch'].to(device)
        label = sample['label'].to(device)
        score_map = model(ref_image, srch_image)
        loss = loss_func(score_map, label)
        loss_val = loss.to('cpu').item()
        avg.update(loss=loss_val, auc=metrics(score_map.detach().cpu().numpy(), label.detach().cpu().numpy()))
        plot_2d_line(avg['loss'](), idx, epoch, len(loader), 'eval', 'loss', viz)
        plot_2d_line(avg['auc'](), idx, epoch, len(loader),'eval', 'auc', viz)
        progbar.set_postfix({"Loss": "%0.4f"%avg['loss'](), "AUC Score": "%0.3f"%avg['auc']()})
        torch.cuda.empty_cache()
    return avg['loss'](), avg['auc']()

def main(args):
    param = Params(args.param_path)
    viz = visdom.Visdom(port=args.port)
    device_num = [int(num) for num in args.gpus.split(',')]
    device = 'cuda' if torch.cuda.is_available() and len(device_num) >= 1 else 'cpu'
    if len(device_num) >= 1:
        torch.cuda.set_device(device_num[0])
    siamfc = SiameseNet(Baseline(), param.corr, param.score_size, param.response_up)
    final_score_sz = siamfc.final_score_sz
    siamfc.apply(weight_init)
    print("Using GPU is {0}\n and".format(device_num), device)
    siamfc = nn.DataParallel(siamfc.to(device), device_ids=device_num).to(device)
    upscale_factor = final_score_sz / param.score_size
    dataset = ImageNetVID(args.root_dir,
                          lable_fcn=create_BCELogit_loss_label,
                          final_size=final_score_sz,
                          pos_thr=param.pos_thr,
                          neg_thr=param.neg_thr,
                          metadata_file=param.train_meta,
                          img_read_fcn=imread,
                          resize_fcn=resize,
                          upscale_factor=upscale_factor,
                          cxt_margin=param.cxt_margin,
                          save_metadata=param.train_meta)
    train_loader = DataLoader(dataset=dataset, batch_size=param.batch_size,
                        shuffle=True, num_workers=param.num_worker, pin_memory=True)

    eval_dataset = ImageNetVID_val(args.root_dir,
                             lable_fcn=create_BCELogit_loss_label,
                                  final_size=final_score_sz,
                                  pos_thr=param.pos_thr,
                                  neg_thr=param.neg_thr,
                                  metadata_file=param.valid_meta,
                                   save_metadata=param.valid_meta,
                                  img_read_fcn=imread,
                                  resize_fcn=resize,
                                  upscale_factor=upscale_factor,
                                  cxt_margin=param.cxt_margin)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.num_worker, pin_memory=True)

    optim = OPTIM[param.optimizer](siamfc.parameters(),
                                   lr=param.optim_param['lr'], momentum=param.optim_param['momentum'], )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, param.lr_decay)
    param.update_with_dict({'start': 0})
    if args.pre_trained !="":
        siamfc, optim, scheduler = load_model(args.pre_trained, siamfc, optim, scheduler, param)
        print(siamfc, optim, scheduler)
        print("Training Resume\n")
    loss_func = BCELogit_Loss
    metrics = AUC

    train_and_evaluate(siamfc, train_loader, eval_loader, optim, loss_func, scheduler, metrics,
                       total_epoch=param.total_epoch, start_epoch=param.start,
                       param=param, device=device, viz=viz)


if __name__ == '__main__':
    arg = parse_arguments()
    main(arg)
