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
    parser.add_argument('-j', '--json-path', required=True, type=str,
                        help="Parameter Json File")
    parser.add_argument('-p', '--port', required=False, type=int, default=8097,
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


def plot_score(viz, drw_data, win=None):
    it_idx = drw_data['iter']
    epoch = drw_data['epoch']
    it = (it_idx+1)*(epoch+1)

    if win is not None:
        win = viz.line(
            X=np.array([it]), Y=np.array([drw_data['loss']]),
            opts=dict(title='{}'.format(drw_data['win_title'])),
            win=win, update='append'
        )
    else:
        win = viz.line(
            X=np.array([it]), Y=np.array([drw_data['loss']]),
            opts=dict(title='{}'.format(drw_data['win_title'])),
            win=win
        )
    return win


def plot_2d_line(vals, iter, epoch, type='train', kinds='loss', viz=None):
    x_axis = (iter+1)*(epoch+1)
    win_name = "{0}_{1}".format(type, kinds)
    viz.line(X=np.array([x_axis]), Y=np.array([vals]), win=win_name, update='append', opts=dict(title=win_name))



def plot_score_n(viz, drw_data, win=None):
    it_idx = drw_data['iter']
    epoch = drw_data['epoch']
    it = (np.array(it_idx)+1)*(epoch+1)

    if win is not None:
        win = viz.line(
            X=np.array(it), Y=np.array(drw_data['loss']),
            opts=dict(title='{}'.format(drw_data['win_title'])),
            win=win, update='append'
        )
    else:
        win = viz.line(
            X=np.array(it), Y=np.array(drw_data['loss']),
            opts=dict(title='{}'.format(drw_data['win_title'])),
            win=win
        )
    return win


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
    for epoch in tqdm(range(start, end)):
        train(model, train_loader, optim, loss_func, metrics,
              epoch=epoch, device=device, viz=viz, display_step=params.display_step)
        # save_model()
        sched.step()
        save_data.update(
            {"model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch}
        )
        save_model(path_to_save=params.chpt_path, save_params=save_data)
        if eval_loader is not None:
            evaluate(model, eval_loader, loss_func, metrics, epoch=epoch)

    print("End Training")


def train(model, loader, optim, loss_func, metrics, device, **kwargs):
    model.train()
    avg = RunningAverageMultiVar(loss=RunningAverage(), auc=RunningAverage())
    progbar = tqdm(loader)
    disp_step = kwargs['display_step']
    viz = kwargs['viz']
    epoch = kwargs['epoch']
    for idx, sample in enumerate(progbar):
        ref_img_batch = sample['ref']
        srch_img_batch = sample['srch']
        label_batch = sample['label'].to(0)
        score_map = data_parallel(model, [ref_img_batch, srch_img_batch], device, 0)
        loss = loss_func(score_map=score_map, labels=label_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_val = loss.to('cpu').item()
        if idx % disp_step == 0 and viz is not None:
            display_images(sample['ref'][0], sample['srch'][0], score_map.to('cpu')[0], sample['label'][0], viz=viz)
        auc_val = metrics(score_map.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
        avg.update(loss=loss_val, auc=auc_val)
        plot_2d_line(loss_val, idx, epoch, type='train', kinds='loss', viz=viz)
        plot_2d_line(auc_val, idx, epoch, type='train', kinds='auc', viz=viz)
        progbar.set_postfix({"Loss": "%0.4f"%avg['loss'](), "AUC Score": "%0.3f"%avg['auc']()})
        torch.cuda.empty_cache()

    return avg['loss'](), avg['auc']()

@torch.no_grad()
def evaluate(model, loader, loss_func, metrics, device, **kwargs):
    epoch = kwargs['epoch']
    model.eval()
    avg = RunningAverageMultiVar(loss=RunningAverage(), auc=RunningAverage())
    for idx, sample in enumerate(loader):
        ref_image = sample['ref']
        srch_image = sample['srch']
        label = sample['label'].to('cuda')
        score_map = data_parallel(model, [ref_image, srch_image], device, 0)
        loss = loss_func(score_map, label)
        loss_val = loss.to('cpu').item()
        avg.update(loss=loss_val, auc=metrics(score_map.detach().cpu().numpy(), label.detach().cpu().numpy()))
        plot_2d_line(avg['loss'](), idx, epoch, 'eval', 'loss')
        plot_2d_line(avg['auc'](), idx, epoch, 'eval', 'auc')
        torch.cuda.empty_cache()
    return avg['loss'](), avg['auc']()


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def main(args):
    param = Params(args.json_path)
    viz = visdom.Visdom(port=args.port)
    device_num = [int(num) for num in args.gpus.split(',')]
    siamfc = SiameseNet(Baseline(), param.corr, param.score_size, param.response_up)
    siamfc = nn.DataParallel(siamfc, device_num, 0)
    siamfc.apply(weight_init)
    upscale_factor = siamfc.module.final_score_sz / param.score_size
    dataset = ImageNetVID(args.root_dir,
                          lable_fcn=create_BCELogit_loss_label,
                          final_size=siamfc.module.final_score_sz,
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
                                  final_size=siamfc.module.final_score_sz,
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
        print("Training Resume\n")
    loss_func = BCELogit_Loss
    metrics = AUC

    train_and_evaluate(siamfc, train_loader, eval_loader, optim, loss_func, scheduler, metrics,
                       total_epoch=param.total_epoch, start_epoch=param.start,
                       param=param, device=device_num, viz=viz)


if __name__ == '__main__':
    arg = parse_arguments()
    main(arg)





