import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_layeredflow_submission(model, iters=24, output_path='layeredflow_submission'):
    """ Create submission for the LayeredFlow leaderboard """
    model.eval()
    test_dataset = datasets.LayeredFlow(downsample=4, split='test', root='datasets/public_layeredflow')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, _, _, _, _ = test_dataset[test_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, f'{test_id}.flo')
        frame_utils.writeFlow(output_filename, flow)

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def validate_layeredflow(model, iters=24):
    """ Peform validation using the LayeredFlow (val) split """
    def datapoint_in_subset(mat, layer, subset):
        def in_list(x, l):
            return l is None or x in l
        assert type(subset) == tuple and len(subset) == 2
        return in_list(mat, subset[0]) and in_list(layer, subset[1])
        
    model.eval()
    val_dataset = datasets.LayeredFlow(downsample=4, split='val')

    subsets = [
        ((0, 2), None), # last layer
        ((0,), None), # last layer, material diffuse
        ((2,), None), # last layer, material reflective
        ((0, 2), (1, 2)), # last layer, at least behind one layer of transparent surface
    ]

    bad_n = [1, 3, 5]
    results = {}
    
    for subset in subsets:
        results[subset] = {}
        results[subset]['epe'] = []
        for n in bad_n:
            results[subset][str(n) + 'px'] = []

    for val_id in range(len(val_dataset)):
        image1, image2, coords, flow_gts, materials, layers = val_dataset[val_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        error_list = {}
        for subset in subsets:
            error_list[subset] = []
        
        for i in range(len(coords)):
            (x, y), mat, lay = coords[i], materials[i], layers[i]

            flow_pd = flow[:, x, y]
            flow_gt = torch.tensor(flow_gts[i])
            error = torch.sum((flow_pd - flow_gt)**2, dim=0).sqrt().item()

            for subset in subsets:
                if datapoint_in_subset(mat, lay, subset):
                    error_list[subset].append(error)

        for subset in subsets:
            if len(error_list[subset]) == 0:
                continue
            error_list[subset] = np.array(error_list[subset])
            results[subset]['epe'].append(np.mean(error_list[subset]))
            for n in bad_n:
                results[subset][str(n) + 'px'].extend(error_list[subset] < n)

    for subset in subsets:
        print(f"Validation LayeredFlow {subset}:")
        for key in results[subset]:
            results[subset][key] = np.mean(results[subset][key])
            if key != 'epe':
                results[subset][key] = 100 - 100 * results[subset][key]
            print(f"{key}: {results[subset][key]}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation", default='layeredflow')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--create_submission', action='store_true', help='create submission')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(os.path.join('checkpoints', args.checkpoint)))

    model.cuda()
    model.eval()

    if args.create_submission:
        if args.dataset == 'sintel':
            create_sintel_submission(model.module, warm_start=True)

        elif args.dataset == 'kitti':
            create_kitti_submission(model.module)
        
        elif args.dataset == 'layeredflow':
            create_layeredflow_submission(model.module, output_path=f'submission_{args.checkpoint}')

    else:
        with torch.no_grad():
            if args.dataset == 'chairs':
                validate_chairs(model.module)

            elif args.dataset == 'sintel':
                validate_sintel(model.module)

            elif args.dataset == 'kitti':
                validate_kitti(model.module)

            elif args.dataset == 'layeredflow':
                validate_layeredflow(model.module)

