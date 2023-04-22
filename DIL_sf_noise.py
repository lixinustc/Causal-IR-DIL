import argparse
import os

import numpy as np
import cv2

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch import distributed as dist
import torch.optim as optim
import srdata_noise
import utils_logger
import logging
import util_calculate_psnr_ssim as util

from RRDB import RRDBNet


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')

    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        "--ckpt_save",
        type=str,
        default=None,
        help="path to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to checkpoints for pretrained model",
    )
    parser.add_argument(
        '--distributed',
        action='store_true'
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument('--trainset', type=str, help='path to the train set')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--testset', type=str, default='default', help='path to the test set, default is Set5')

    parser.add_argument('--save_every', type=int, default=1, help='save weights')
    parser.add_argument('--test_every', type=int, default=5, help='save weights')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--num_workder', type=int, default=8)
    parser.add_argument('--total_epoch', type=int, default=30)

    args = parser.parse_args()

    return args

def data_sampler(dataset, shuffle=True, distributed=True):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def point_grad_to(meta_net, task_net):
    '''
    Set .grad attribute of each parameter to be proportional
    to the difference between self and target
    '''
    for meta_p, task_p in zip(meta_net.parameters(), task_net.parameters()):
        if meta_p.grad is None:
            meta_p.grad = torch.zeros(meta_p.size()).cuda()
        # meta_p.grad.data.zero_()  # not required as optimizer.zero_grad
        meta_p.grad.data.add_(meta_p.data - task_p.data)


def main():
    
    args = parse_args()

    ## initialize training folder
    checkpoint_save_path = args.ckpt_save
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path, exist_ok=True)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(checkpoint_save_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    ## initialize DDP training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ## initialize model and optimizer
    model_task = RRDBNet(in_nc=3, out_nc=3).to('cuda')
    model_meta = RRDBNet(in_nc=3, out_nc=3).to('cuda')

    optimizer_task = optim.Adam([p for p in model_task.parameters() if p.requires_grad], lr=1.e-4, betas=(0, 0.999))
    optimizer_meta = optim.Adam([p for p in model_meta.parameters() if p.requires_grad], lr=1.e-4)

    if args.resume is not None:
        print("load model: ", args.resume)
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model_task.load_state_dict(ckpt['model_task'])
        model_meta.load_state_dict(ckpt['model_meta'])

    
    loss_fn = torch.nn.L1Loss()
    loss_fn = loss_fn.to('cuda')

    ## for gaussian denoising, we set task number to 4
    dataset_list = []
    for i in range(4):
        dataset_list.append(srdata_noise.DataCrop(i, hr_folder=args.trainset, patch_size=args.patch_size))

    testset = srdata_noise.DataTest(hr_folder=args.testset, level=50)  # you can try 50, 70 ...

    dataloader_test = data.DataLoader(
        testset, 
        batch_size=1,
        sampler=data_sampler(testset, shuffle=False, distributed=False),
        num_workers=1,
        pin_memory=True
    )

    dataloader_list = [
        data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        sampler=data_sampler(trainset, shuffle=True, distributed=args.distributed),
        num_workers=args.num_workder,
        pin_memory=True,
        drop_last=True
        )
        for trainset in dataset_list
    ]


    if args.distributed:
        model_task = DistributedDataParallel(
            model_task,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )
        model_meta = DistributedDataParallel(
            model_meta,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )


    total_epochs = args.total_epoch
    state_task = None

    for epoch in range(total_epochs):

        if epoch and not (epoch % 25):
            for param in optimizer_meta.param_groups:
                param['lr'] = (param['lr'] * 0.5) if param['lr'] > 1.e-6 else 1.e-6
            sche = True

        learning_rate_f = optimizer_task.param_groups[0]['lr']
        learning_rate_s = optimizer_meta.param_groups[0]['lr']

        data_len = len(dataset_list[0])

        data_loader_train = [iter(dataloader) for dataloader in dataloader_list]

        random_list = [0, 1, 2, 3]
        np.random.seed(1)  # to control random_list is same on every gpus.

        for iteration in range(data_len // (args.batch_size * args.gpus)):
            model_task.load_state_dict(model_meta.state_dict())

            if state_task is not None:
                optimizer_task.load_state_dict(state_task)

            np.random.shuffle(random_list)

            for ind in random_list:
                dl = data_loader_train[ind]

                lr, hr = dl.next()
                    
                optimizer_task.zero_grad()
                lr = lr.to('cuda')
                hr = hr.to('cuda')
                sr = model_task(lr)
                loss = loss_fn(sr, hr)
                loss_print = loss.item()
                loss.backward()
                optimizer_task.step()

                if torch.cuda.current_device() == 0 and not iteration % args.print_every:
                    logger.info('Epoch: {}\tIter: {}/{}\tTask loss: {}\tTask LR: {:.6f}\tMeta LR: {:.6f}'.format(epoch, iteration, data_len // (args.batch_size * args.gpus), loss_print, learning_rate_f, learning_rate_s))

            state_task = optimizer_task.state_dict()
            optimizer_meta.zero_grad()
            point_grad_to(model_meta, model_task)
            optimizer_meta.step()
            
            if torch.cuda.current_device() == 0 and not iteration % args.print_every:
                logger.info('Meta net updated!')

        # save model
        if not epoch % args.save_every and torch.cuda.current_device() == 0:
            m_task = model_task.module if args.distributed else model_task
            m_meta = model_meta.module if args.distributed else model_meta
            model_meta_dict = m_meta.state_dict()
            model_task_dict = m_task.state_dict()
            torch.save(
                {
                    'model_meta': model_meta_dict,
                    'model_task': model_task_dict,
                },
                os.path.join(checkpoint_save_path, 'model_{}.pt'.format(epoch+1))
            )
        # test model
        if not epoch % args.test_every and torch.cuda.current_device() == 0:
                model_meta.eval()
                p = 0
                s = 0
                count = 0
                
                for lr, hr, filename in dataloader_test:
                    count += 1
                    lr = lr.to('cuda')
                    filename = filename[0]
                    with torch.no_grad():
                        sr = model_meta(lr)
                    sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                    sr = sr * 255.
                    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
                    hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
                    hr = hr * 255.
                    hr = np.clip(hr.round(), 0, 255).astype(np.uint8)

                    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
                    psnr = util.calculate_psnr(sr, hr, crop_border=0)
                    ssim = util.calculate_ssim(sr, hr, crop_border=0)
                    p += psnr
                    s += ssim
                    logger.info('{}: {}, {}'.format(filename, psnr, ssim))

                p /= count
                s /= count
                logger.info("Epoch: {}, psnr: {}. ssim: {}.".format(epoch, p, s))
                
                model_meta.train()
    
    
    logger.info('Done')

if __name__ == '__main__':
    main()