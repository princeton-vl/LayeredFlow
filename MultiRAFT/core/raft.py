import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class MultiRAFT(nn.Module):
    def __init__(self, args):
        super(MultiRAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnets = nn.ModuleList([SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout) for _ in range(4)])
            self.cnets = nn.ModuleList([SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout) for _ in range(4)])
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.duplications = args.duplications
            if 'fnet' in args.duplications:
                self.fnets = nn.ModuleList([BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout) for _ in range(4)])    
            else:
                self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)

            if 'cnet' in args.duplications:
                self.cnets = nn.ModuleList([BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout) for _ in range(4)])
            else:
                self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            
            if 'update_block' in args.duplications:
                self.update_blocks = nn.ModuleList([BasicUpdateBlock(self.args, hidden_dim=hdim) for _ in range(4)])
            else:
                self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

            # self.mask_prediction = MaskPredBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
    
    def upsample_mask(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1s = []
        fmap2s = []
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            if 'fnet' in self.duplications:
                for fnet in self.fnets:
                    fmap1, fmap2 = fnet([image1, image2])
                    fmap1s.append(fmap1.float())
                    fmap2s.append(fmap2.float())
            else:
                fmap1, fmap2 = self.fnet([image1, image2])
                fmap1, fmap2 = fmap1.float(), fmap2.float()    
                fmap1s = [fmap1.clone() for _ in range(4)]
                fmap2s = [fmap2.clone() for _ in range(4)]

        corr_fns = []
        if 'fnet' in self.duplications:
            for fmap1, fmap2 in zip(fmap1s, fmap2s):
                if self.args.alternate_corr:
                    corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
                else:
                    corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
                corr_fns.append(corr_fn)
        else:
            if self.args.alternate_corr:
                corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            else:
                corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            corr_fns = [corr_fn for _ in range(4)]

        # run the context network
        nets = []
        inps = []
        with autocast(enabled=self.args.mixed_precision):
            if 'cnet' in self.duplications:
                for cnet in self.cnets:
                    output = cnet(image1)
                    net, inp = torch.split(output, [hdim, cdim], dim=1)
                    net = torch.tanh(net)
                    inp = torch.relu(inp)
                    nets.append(net)
                    inps.append(inp)
            else:
                output = self.cnet(image1)
                net, inp = torch.split(output, [hdim, cdim], dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)
                nets = [net.clone() for _ in range(4)]
                inps = [inp.clone() for _ in range(4)]
 
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        # Extend the coords to four layers
        coords1s = [coords1.clone().detach() for _ in range(4)]

        all_flow_predictions = []
        for layer in range(4):
            coords1 = coords1s[layer]
            corr_fn = corr_fns[layer]
            net = nets[layer]
            inp = inps[layer]

            # if layer > 0:
            #     coords1 = coords1 + (coords1s[layer-1] - coords0)

            flow_predictions = []
            for itr in range(iters):    
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    if 'update_block' in self.duplications:
                        net, up_mask, delta_flow = self.update_blocks[layer](net, inp, corr, flow)
                    else:
                        net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow

                # upsample predictions
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                
                flow_predictions.append(flow_up)

            flow_predictions = torch.stack(flow_predictions, dim=1)
            all_flow_predictions.append(flow_predictions)
        
        all_flow_predictions = torch.stack(all_flow_predictions, dim=1)
  
        if test_mode:
            flow_predictions = all_flow_predictions[:, :, -1]
            # b, n_layer, n_flow, h, w = flow_predictions.shape
            # mask_predictions = torch.zeros(b, 1, h, w).cuda()

            # layer0_layer1_offset = flow_predictions[:, 0] - flow_predictions[:, 1]
            # layer1_layer2_offset = flow_predictions[:, 1] - flow_predictions[:, 2]
            # layer2_layer3_offset = flow_predictions[:, 2] - flow_predictions[:, 3]

            # layer0_layer1_offset = torch.sum(layer0_layer1_offset ** 2, dim=1).sqrt()
            # layer1_layer2_offset = torch.sum(layer1_layer2_offset ** 2, dim=1).sqrt()
            # layer2_layer3_offset = torch.sum(layer2_layer3_offset ** 2, dim=1).sqrt()
            
            # mask_predictions += layer0_layer1_offset > 0.5
            # mask_predictions += layer1_layer2_offset > 0.5
            # mask_predictions += layer2_layer3_offset > 0.5

            return flow_predictions #, mask_predictions
            
        return all_flow_predictions #, all_mask_predictions