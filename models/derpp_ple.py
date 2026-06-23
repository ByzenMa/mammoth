"""DER++ with Progressive Layered Extraction feature fusion."""

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
    return names or [fallback]


def _parse_hidden_units(hidden_units: str) -> List[int]:
    if hidden_units is None or hidden_units.strip() == '':
        return []
    return [int(unit.strip()) for unit in hidden_units.split(',') if unit.strip()]


def _activation(name: str) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Unsupported PLE activation `{name}`.')


def _make_mlp(hidden_units: Sequence[int], dropout: float, activation: str, final_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for hidden_dim in hidden_units:
        layers.extend([nn.LazyLinear(hidden_dim), _activation(activation)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.LazyLinear(final_dim))
    layers.append(_activation(activation))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _make_gate(hidden_units: Sequence[int], dropout: float, activation: str, output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for hidden_dim in hidden_units:
        layers.extend([nn.LazyLinear(hidden_dim), _activation(activation)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.extend([nn.LazyLinear(output_dim), nn.Softmax(dim=1)])
    return nn.Sequential(*layers)


class PLEBlock(nn.Module):
    """Customized Gate Control block for Progressive Layered Extraction."""

    def __init__(self, num_tasks: int, shared_expert_num: int, specific_expert_num: int, expert_hidden_units: Sequence[int],
                 gate_hidden_units: Sequence[int], expert_dim: int, dropout: float, activation: str, is_last: bool) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.is_last = is_last
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([_make_mlp(expert_hidden_units, dropout, activation, expert_dim) for _ in range(specific_expert_num)])
            for _ in range(num_tasks)
        ])
        self.shared_experts = nn.ModuleList([_make_mlp(expert_hidden_units, dropout, activation, expert_dim) for _ in range(shared_expert_num)])
        self.task_gates = nn.ModuleList([
            _make_gate(gate_hidden_units, dropout, activation, specific_expert_num + shared_expert_num)
            for _ in range(num_tasks)
        ])
        if not is_last:
            self.shared_gate = _make_gate(gate_hidden_units, dropout, activation, num_tasks * specific_expert_num + shared_expert_num)

    @staticmethod
    def _gate_experts(experts: Sequence[torch.Tensor], gate: torch.Tensor) -> torch.Tensor:
        expert_stack = torch.stack(list(experts), dim=1)
        return torch.sum(expert_stack * gate.unsqueeze(-1), dim=1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        specific_outputs = [
            [expert(inputs[task_idx]) for expert in self.specific_experts[task_idx]]
            for task_idx in range(self.num_tasks)
        ]
        shared_outputs = [expert(inputs[-1]) for expert in self.shared_experts]
        outputs = []
        for task_idx in range(self.num_tasks):
            outputs.append(self._gate_experts(specific_outputs[task_idx] + shared_outputs, self.task_gates[task_idx](inputs[task_idx])))
        if not self.is_last:
            shared_experts = [expert for task_experts in specific_outputs for expert in task_experts] + shared_outputs
            outputs.append(self._gate_experts(shared_experts, self.shared_gate(inputs[-1])))
        return outputs


class PLEBackbone(nn.Module):
    """Fuse one or more backbone feature vectors with PLE and emit classification logits."""

    def __init__(self, backbones: Sequence[nn.Module], num_classes: int, ple_num_tasks: int, ple_num_levels: int,
                 ple_shared_expert_num: int, ple_specific_expert_num: int, ple_expert_dim: int,
                 ple_expert_hidden_units: str, ple_gate_hidden_units: str, ple_tower_hidden_units: str,
                 ple_dropout: float, ple_activation: str, ple_output_mode: str, ple_output_index: int) -> None:
        super().__init__()
        if len(backbones) == 0:
            raise ValueError('PLEBackbone requires at least one backbone.')
        if ple_num_tasks < 1:
            raise ValueError('--ple_num_tasks must be at least 1.')
        if ple_num_levels < 1:
            raise ValueError('--ple_num_levels must be at least 1.')
        if ple_output_mode not in ['mean', 'target']:
            raise ValueError('--ple_output_mode must be `mean` or `target`.')
        if ple_output_mode == 'target' and not 0 <= ple_output_index < ple_num_tasks:
            raise ValueError(f'--ple_output_index must be in [0, {ple_num_tasks - 1}] for {ple_num_tasks} PLE targets.')
        self.backbones = nn.ModuleList(backbones)
        self.num_tasks = ple_num_tasks
        self.output_mode = ple_output_mode
        self.output_index = ple_output_index
        expert_hidden_units = _parse_hidden_units(ple_expert_hidden_units)
        gate_hidden_units = _parse_hidden_units(ple_gate_hidden_units)
        tower_hidden_units = _parse_hidden_units(ple_tower_hidden_units)
        self.ple_blocks = nn.ModuleList([
            PLEBlock(ple_num_tasks, ple_shared_expert_num, ple_specific_expert_num, expert_hidden_units, gate_hidden_units,
                     ple_expert_dim, ple_dropout, ple_activation, is_last=level_idx == ple_num_levels - 1)
            for level_idx in range(ple_num_levels)
        ])
        self.towers = nn.ModuleList([_make_mlp(tower_hidden_units, ple_dropout, ple_activation, ple_expert_dim) for _ in range(ple_num_tasks)])
        self.heads = nn.ModuleList([nn.LazyLinear(num_classes) for _ in range(ple_num_tasks)])

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = [backbone(x, returnt='features') for backbone in self.backbones]
        return torch.cat([feature.flatten(1) for feature in features], dim=1)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        features = self._extract_features(x)
        ple_inputs: List[torch.Tensor] = [features] * (self.num_tasks + 1)
        for block in self.ple_blocks:
            ple_inputs = block(ple_inputs)
        tower_features = [tower(ple_inputs[task_idx]) for task_idx, tower in enumerate(self.towers)]
        task_logits = torch.stack([head(tower_features[task_idx]) for task_idx, head in enumerate(self.heads)], dim=1)
        logits = task_logits.mean(dim=1) if self.output_mode == 'mean' else task_logits[:, self.output_index]
        if returnt in ('out', 'logits'):
            return logits
        if returnt in ('both', 'full', 'all'):
            return logits, task_logits
        if returnt == 'features':
            return torch.cat(tower_features, dim=1)
        raise ValueError(f'Unsupported returnt value for PLEBackbone: {returnt}')


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


class DerppPle(ContinualModel):
    """DER++ with PLE fusion over configurable backbone features."""

    NAME = 'derpp-ple'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight for DER++ logit replay.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight for DER++ label replay.')
        parser.add_argument('--ple_backbones', type=str, default=None,
                            help='Comma-separated backbone names whose features are fused by PLE. Defaults to the selected --backbone.')
        parser.add_argument('--ple_num_tasks', type=int, default=2,
                            help='Number of PLE task/target branches.')
        parser.add_argument('--ple_num_levels', type=int, default=2,
                            help='Number of stacked PLE extraction levels.')
        parser.add_argument('--ple_shared_expert_num', type=int, default=1,
                            help='Number of shared experts in each PLE level.')
        parser.add_argument('--ple_specific_expert_num', type=int, default=1,
                            help='Number of task-specific experts per PLE target in each level.')
        parser.add_argument('--ple_expert_dim', type=int, default=128,
                            help='Output dimension of each PLE expert and tower.')
        parser.add_argument('--ple_expert_hidden_units', type=str, default='256',
                            help='Comma-separated hidden units for each PLE expert MLP; empty means only the expert output layer.')
        parser.add_argument('--ple_gate_hidden_units', type=str, default='',
                            help='Comma-separated hidden units for each PLE gate MLP; empty means a linear softmax gate.')
        parser.add_argument('--ple_tower_hidden_units', type=str, default='64',
                            help='Comma-separated hidden units for each PLE tower MLP; empty means only the tower output layer.')
        parser.add_argument('--ple_dropout', type=float, default=0.0,
                            help='Dropout probability used inside PLE experts, gates, and towers.')
        parser.add_argument('--ple_activation', type=str, default='relu', choices=['relu', 'gelu', 'tanh'],
                            help='Activation function used inside PLE experts, gates, and towers.')
        parser.add_argument('--ple_output_mode', type=str, default='mean', choices=['mean', 'target'],
                            help='How to convert PLE target logits into final logits: average all targets or select one target branch.')
        parser.add_argument('--ple_output_index', type=int, default=0,
                            help='PLE target branch index used when --ple_output_mode target.')
        parser.add_argument('--clip_model_name', type=str, default='ViT-B-16',
                            help='CLIP architecture name used when `clip` is listed in --ple_backbones.')
        parser.add_argument('--clip_checkpoint_path', type=str, default=None,
                            help='Local CLIP checkpoint path used when `clip` is listed in --ple_backbones.')
        parser.add_argument('--freeze_clip', type=int, default=0,
                            help='Freeze the CLIP visual encoder when `clip` is listed in --ple_backbones.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone_names = _parse_backbone_names(args.ple_backbones, args.backbone)
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
            logging.warning('Primary --backbone `%s` is not listed in --ple_backbones and will not be used.', args.backbone)

        ple_backbone = PLEBackbone(
            backbones, num_classes, args.ple_num_tasks, args.ple_num_levels, args.ple_shared_expert_num,
            args.ple_specific_expert_num, args.ple_expert_dim, args.ple_expert_hidden_units,
            args.ple_gate_hidden_units, args.ple_tower_hidden_units, args.ple_dropout,
            args.ple_activation, args.ple_output_mode, args.ple_output_index)
        super().__init__(ple_backbone, loss, args, transform, dataset=dataset)
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
