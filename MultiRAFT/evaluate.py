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
from tqdm import tqdm
import torch.utils.data as data
from time import sleep

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import MultiRAFT
from utils.utils import InputPadder, forward_interpolate

@torch.no_grad()
def create_layeredflow_submission(model, iters=24, output_path='layeredflow_submission'):
    """ Create submission for the LayeredFlow leaderboard """
    model.eval()
    test_dataset = datasets.LayeredFlow(downsample=4, split='test')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = {}
    for test_id in range(len(test_dataset)):
        image1, image2, coords, _, _, _ = test_dataset[test_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_prs = model(image1, image2, iters=iters, test_mode=True)
        flows = padder.unpad(flow_prs)[0].cpu().numpy()  # Convert to numpy array for processing

        h, w = flows.shape[-2:]
        flows_upd = np.full((4, 2, h, w), np.nan)  # Initialize with NaN

        flows_reshaped = flows.reshape(4, 2, h * w)  # Shape (4, 2, h * w)
        flows_pred, layer_pred = vectorized_get_layer_pred(flows_reshaped)
        flows_pred = flows_pred.reshape(4, 2, h, w)  # Reshape back to original shape

        sorted_indices = np.argsort(np.isnan(flows_pred), axis=0)
        flows_pred = np.take_along_axis(flows_pred, sorted_indices, axis=0)
        flows_upd = flows_pred
        results[str(test_id)] = flows_upd

    output_filename = os.path.join(output_path, 'results.npz')
    np.savez(output_filename, **results)

def get_layer_pred(flow):
    assert flow.shape[0] == 4

    for i in range(4):
        if np.isnan(flow[i, 0]):
            return flow[:i], i - 1
    return flow, 3

def vectorized_get_layer_pred(flows):
    threshold = 0.5
    flows = np.array(flows)
    distance01 = np.sqrt(np.sum((flows[0] - flows[1])**2, axis=0))  # Shape (h * w,)
    distance12 = np.sqrt(np.sum((flows[1] - flows[2])**2, axis=0))  # Shape (h * w,)
    distance23 = np.sqrt(np.sum((flows[2] - flows[3])**2, axis=0))  # Shape (h * w,)

    layer_pred = (distance01 > threshold).astype(int) + (distance12 > threshold).astype(int) + (distance23 > threshold).astype(int)

    flow_mask = np.array([np.ones(flows.shape[-1], dtype=bool),
                          distance01 > threshold,
                          distance12 > threshold,
                          distance23 > threshold])

    flow_pred = np.where(flow_mask[:, None, :], flows, np.nan)    
    return flow_pred, layer_pred

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
        (None, (0,)), # first layer
        (None, (1,)), # second layer
        (None, (2,)), # third layer
    ]

    bad_n = [1, 3, 5, float('inf')]
    results = {}
    
    for subset in subsets:
        results[subset] = {}
        for n in bad_n:
            results[subset][str(n) + 'px'] = []
    
    for val_id in range(len(val_dataset)):
        image1, image2, coords, flow_gts, materials, layers = val_dataset[val_id]
        image1, image2 = image1[None].cuda(), image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_prs = model(image1, image2, iters=iters, test_mode=True)
        flows = padder.unpad(flow_prs)[0].cpu()

        h, w = flows.shape[-2:]
        flows_upd = np.full((4, 2, h, w), np.nan)
        flows_reshaped = flows.reshape(4, 2, h * w)

        flows_pred, layer_pred = vectorized_get_layer_pred(flows_reshaped)
        flows_pred = flows_pred.reshape(4, 2, h, w)  

        sorted_indices = np.argsort(np.isnan(flows_pred), axis=0)
        flows_pred = np.take_along_axis(flows_pred, sorted_indices, axis=0)
        flows_upd = flows_pred

        for i in range(len(coords)):
            (x, y), mat, lay = coords[i], materials[i], layers[i]
            flow_pd, layer_pd = get_layer_pred(flows_upd[:, :, x, y])
            flow_pd = torch.tensor(flow_pd).float()

            # Check layer correctness
            if mat == 1:
                layer_correct = (layer_pd >= lay)
            else:
                layer_correct = (layer_pd == lay)
            
            # Error in pixel
            if layer_correct:
                flow_gt = torch.tensor([flow_gts[i][0].item(), flow_gts[i][1].item()]).float()
                flow_pd = flow_pd[lay]
                error = torch.sum((flow_pd - flow_gt)**2).sqrt().item()
            else:
                error = float('inf')

            for subset in subsets:
                if datapoint_in_subset(mat, lay, subset):
                    for n in bad_n:
                        if layer_correct and (n == float('inf') or error < n):
                            results[subset][str(n) + 'px'].append(1)
                        else:
                            results[subset][str(n) + 'px'].append(0)

    for subset in subsets:
        print(f"Validation LayeredFlow {subset}:")
        for key in results[subset]:
            results[subset][key] = np.mean(results[subset][key])
            results[subset][key] = 100 - 100 * results[subset][key]
            print(f"{key}: {results[subset][key]}")
    return results

def load_pretrain(args, model):
    load_pretrained = os.path.join('checkpoints', 'raft-sintel.pth')
    curr_dict = model.state_dict()
    pretrained_dict = torch.load(load_pretrained)
    for key in list(pretrained_dict.keys()):
        for net_name in args.duplications:
            if net_name in key:
                key_decomposed = key.split(net_name)
                for idx in range(4):
                    new_key = f'{net_name}s.{idx}'.join(key_decomposed)
                    pretrained_dict[new_key] = pretrained_dict[key]
                del pretrained_dict[key]
            
    curr_dict.update(pretrained_dict)
    model.load_state_dict(curr_dict)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation", default='layeredflow')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--duplications', type=str, nargs='+', default=['cnet'])
    parser.add_argument('--create_submission', action='store_true', help='create submission for LayeredFlow')
    args = parser.parse_args()
    # val_dataset = datasets.TransBench(downsample=4)

    model = torch.nn.DataParallel(MultiRAFT(args))
    
    if args.checkpoint != 'raft-sintel.pth':
        model.load_state_dict(torch.load(os.path.join('checkpoints', args.checkpoint)))
    else:
        model = load_pretrain(args, model)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'layeredflow':
            if args.create_submission:
                create_layeredflow_submission(model)
            else:
                validate_layeredflow(model.module)

