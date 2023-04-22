import argparse
import os
import torch

import cv2
import numpy as np
from collections import OrderedDict
from RRDB import RRDBNet
import util_calculate_psnr_ssim as util

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument("--ckpt", type=str, default=None, help="path to load checkpoint",)
parser.add_argument('--testset', type=str, help='path to the test set')
parser.add_argument('--save', type=str, help='path to save results')
parser.add_argument('--level', type=int, default=50)

args = parser.parse_args()

assert os.path.isfile(args.ckpt), 'please check your checkpoint path!'

weight = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
weight = weight['model_meta']

model = RRDBNet(in_nc=3, out_nc=3).to('cuda')
model.load_state_dict(weight)
model = model.cuda()

test_path = args.testset
output_path = args.save

if not os.path.exists(output_path):
    os.makedirs(output_path)

f = open(os.path.join(output_path, 'log.txt'),'w')

model.eval()

count = 0
p = 0
s = 0

for img_n in sorted(os.listdir(test_path)):
    count += 1

    hr = cv2.imread(os.path.join(test_path, img_n))
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)  # RGB, n_channels=3
    
    # generate lr online
    lr = hr.copy()
    lr = lr.astype(np.float32) / 255.
    noise = np.random.randn(*lr.shape) * args.level / 255.
    lr += noise
    lr = np.clip(lr, 0, 1).astype(np.float32)
    lr = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1)))
    lr = lr.unsqueeze(0).cuda()
    
    with torch.no_grad():
        lr = model(lr)

    sr = lr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)

    sr = sr * 255.
    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)

    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)

    psnr = util.calculate_psnr(hr, sr, crop_border=0)
    ssim = util.calculate_ssim(hr, sr, crop_border=0)

    p += psnr
    s + ssim
    f.write('{}: PSNR, {}; SSIM, {}.\n'.format(img_n, psnr, ssim))

    cv2.imwrite(os.path.join(output_path, img_n), sr)

p /= count
s /= count

f.write('avg PSNR: {}, avg SSIM: {}.'.format(p, s))

f.close()
