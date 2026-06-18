"""DER++ with a linear-attention ensemble over multiple backbones."""

import copy
import logging
from argparse import Namespace
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone import get_backbone_class
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


def _parse_backbone_names(backbones: str, fallback: str) -> List[str]:
    if backbones is None or backbones.strip() == '':
        return [fallback]
    names = [name.strip().replace('_', '-').lower() for name in backbones.split(',') if name.strip()]
    if not names:
        return [fallback]
    return names


class LinearAttentionBackbone(nn.Module):
    """Fuse multiple classifier backbones with a learned linear attention gate."""

    def __init__(self, backbones: Sequence[nn.Module], num_classes: int) -> None:
        super().__init__()
        if len(backbones) == 0:
            raise ValueError('LinearAttentionBackbone requires at least one backbone.')
        self.backbones = nn.ModuleList(backbones)
        self.attention = nn.Linear(num_classes, 1)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        outputs = torch.stack([backbone(x) for backbone in self.backbones], dim=1)
        weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
        fused = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        if returnt in ('out', 'logits'):
            return fused
        if returnt in ('both', 'full', 'all'):
            return fused, outputs
        if returnt == 'features':
            return outputs.flatten(1)
        raise ValueError(f'Unsupported returnt value for LinearAttentionBackbone: {returnt}')


def _build_registered_backbone(name: str, args: Namespace, num_classes: int) -> nn.Module:
    backbone_class, backbone_args = get_backbone_class(name, return_args=True)
    parsed_args = {}
    for arg_name, arg_conf in backbone_args.items():
        if arg_name == 'num_classes':
            parsed_args[arg_name] = num_classes
        elif hasattr(args, arg_name):
            parsed_args[arg_name] = getattr(args, arg_name)
        elif arg_conf.get('required', False):
            raise ValueError(f'Missing required argument `{arg_name}` for auxiliary backbone `{name}`.')
        else:
            parsed_args[arg_name] = arg_conf.get('default')
    return backbone_class(**parsed_args)


class DerppLinearAttention(ContinualModel):
    """DER++ with linearly-attended logits from multiple configurable backbones."""

    NAME = 'derpp-linear-attention'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight for DER++ logit replay.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight for DER++ label replay.')
        parser.add_argument('--attention_backbones', type=str, default=None,
                            help='Comma-separated backbone names to fuse with linear attention. Defaults to the selected --backbone.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone_names = _parse_backbone_names(args.attention_backbones, args.backbone)
        num_classes = dataset.N_CLASSES if dataset is not None else args.n_classes
        backbones = []
        used_primary = False
        for name in backbone_names:
            if name == args.backbone.replace('_', '-').lower() and not used_primary:
                backbones.append(backbone)
                used_primary = True
            else:
                logging.info('Building auxiliary backbone `%s` for %s.', name, self.NAME)
                backbones.append(_build_registered_backbone(name, copy.copy(args), num_classes))
        if not used_primary:
            logging.warning('Primary --backbone `%s` is not listed in --attention_backbones and will not be used.', args.backbone)

        fused_backbone = LinearAttentionBackbone(backbones, num_classes)
        super().__init__(fused_backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()
