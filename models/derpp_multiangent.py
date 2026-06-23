"""DER++ with independently configurable multi-backbone and MultiAngent fusion."""

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


def _normalize_backbone_name(name: str) -> str:
    return name.strip().replace('_', '-').lower()


def _parse_backbone_names(backbones: str, fallback: str) -> List[str]:
    if backbones is None or backbones.strip() == '':
        return [_normalize_backbone_name(fallback)]
    names = [_normalize_backbone_name(name) for name in backbones.split(',') if name.strip()]
    return names or [_normalize_backbone_name(fallback)]


def _parse_fusion_weights(weights: str, n_backbones: int) -> List[float]:
    if weights is None or weights.strip() == '':
        return [1.0 / n_backbones] * n_backbones
    parsed_weights = [float(weight.strip()) for weight in weights.split(',') if weight.strip()]
    if len(parsed_weights) != n_backbones:
        raise ValueError(f'Expected {n_backbones} fusion weights, got {len(parsed_weights)} from `{weights}`.')
    weight_sum = sum(parsed_weights)
    if weight_sum <= 0:
        raise ValueError('Manual fusion weights must have a positive sum.')
    return [weight / weight_sum for weight in parsed_weights]


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
    raise ValueError(f'Unsupported MultiAngent activation `{name}`.')


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


class LogitFusionBackbone(nn.Module):
    """Fuse one or more classifier backbones at logits level."""

    def __init__(self, backbones: Sequence[nn.Module], num_classes: int, fusion_mode: str = 'linear_attention', fusion_weights: str = None) -> None:
        super().__init__()
        if len(backbones) == 0:
            raise ValueError('LogitFusionBackbone requires at least one backbone.')
        if fusion_mode not in ['linear_attention', 'manual']:
            raise ValueError(f'Unsupported fusion mode `{fusion_mode}`. Use `linear_attention` or `manual`.')
        self.backbones = nn.ModuleList(backbones)
        self.fusion_mode = fusion_mode
        if self.fusion_mode == 'linear_attention':
            self.attention = nn.Linear(num_classes, 1)
        else:
            weights = torch.tensor(_parse_fusion_weights(fusion_weights, len(backbones)), dtype=torch.float32)
            self.register_buffer('manual_weights', weights.view(1, len(backbones), 1), persistent=True)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        outputs = torch.stack([backbone(x) for backbone in self.backbones], dim=1)
        if self.fusion_mode == 'linear_attention':
            weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1).unsqueeze(-1)
        else:
            weights = self.manual_weights.to(device=outputs.device, dtype=outputs.dtype)
        fused = torch.sum(outputs * weights, dim=1)
        if returnt in ('out', 'logits'):
            return fused
        if returnt in ('both', 'full', 'all'):
            return fused, outputs
        if returnt == 'features':
            return outputs.flatten(1)
        raise ValueError(f'Unsupported returnt value for LogitFusionBackbone: {returnt}')


class MultiAngentBlock(nn.Module):
    """Multi-target shared/specific expert block inspired by CGC-style extraction."""

    def __init__(self, num_targets: int, shared_expert_num: int, specific_expert_num: int, expert_hidden_units: Sequence[int],
                 gate_hidden_units: Sequence[int], expert_dim: int, dropout: float, activation: str, is_last: bool) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.is_last = is_last
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([_make_mlp(expert_hidden_units, dropout, activation, expert_dim) for _ in range(specific_expert_num)])
            for _ in range(num_targets)
        ])
        self.shared_experts = nn.ModuleList([_make_mlp(expert_hidden_units, dropout, activation, expert_dim) for _ in range(shared_expert_num)])
        self.target_gates = nn.ModuleList([
            _make_gate(gate_hidden_units, dropout, activation, specific_expert_num + shared_expert_num)
            for _ in range(num_targets)
        ])
        if not is_last:
            self.shared_gate = _make_gate(gate_hidden_units, dropout, activation, num_targets * specific_expert_num + shared_expert_num)

    @staticmethod
    def _gate_experts(experts: Sequence[torch.Tensor], gate: torch.Tensor) -> torch.Tensor:
        expert_stack = torch.stack(list(experts), dim=1)
        return torch.sum(expert_stack * gate.unsqueeze(-1), dim=1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        specific_outputs = [
            [expert(inputs[target_idx]) for expert in self.specific_experts[target_idx]]
            for target_idx in range(self.num_targets)
        ]
        shared_outputs = [expert(inputs[-1]) for expert in self.shared_experts]
        outputs = []
        for target_idx in range(self.num_targets):
            outputs.append(self._gate_experts(specific_outputs[target_idx] + shared_outputs, self.target_gates[target_idx](inputs[target_idx])))
        if not self.is_last:
            shared_experts = [expert for target_experts in specific_outputs for expert in target_experts] + shared_outputs
            outputs.append(self._gate_experts(shared_experts, self.shared_gate(inputs[-1])))
        return outputs


class MultiAngentBackbone(nn.Module):
    """Fuse backbone features with MultiAngent expert/gate layers and emit logits."""

    def __init__(self, backbones: Sequence[nn.Module], num_classes: int, num_targets: int, num_levels: int,
                 shared_expert_num: int, specific_expert_num: int, expert_dim: int,
                 expert_hidden_units: str, gate_hidden_units: str, tower_hidden_units: str,
                 dropout: float, activation: str, output_mode: str, output_index: int) -> None:
        super().__init__()
        if len(backbones) == 0:
            raise ValueError('MultiAngentBackbone requires at least one backbone.')
        if num_targets < 1:
            raise ValueError('--multiangent_num_targets must be at least 1.')
        if num_levels < 1:
            raise ValueError('--multiangent_num_levels must be at least 1.')
        if output_mode not in ['mean', 'target']:
            raise ValueError('--multiangent_output_mode must be `mean` or `target`.')
        if output_mode == 'target' and not 0 <= output_index < num_targets:
            raise ValueError(f'--multiangent_output_index must be in [0, {num_targets - 1}] for {num_targets} targets.')
        self.backbones = nn.ModuleList(backbones)
        self.num_targets = num_targets
        self.output_mode = output_mode
        self.output_index = output_index
        expert_hidden = _parse_hidden_units(expert_hidden_units)
        gate_hidden = _parse_hidden_units(gate_hidden_units)
        tower_hidden = _parse_hidden_units(tower_hidden_units)
        self.blocks = nn.ModuleList([
            MultiAngentBlock(num_targets, shared_expert_num, specific_expert_num, expert_hidden, gate_hidden,
                             expert_dim, dropout, activation, is_last=level_idx == num_levels - 1)
            for level_idx in range(num_levels)
        ])
        self.towers = nn.ModuleList([_make_mlp(tower_hidden, dropout, activation, expert_dim) for _ in range(num_targets)])
        self.heads = nn.ModuleList([nn.LazyLinear(num_classes) for _ in range(num_targets)])

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = [backbone(x, returnt='features') for backbone in self.backbones]
        return torch.cat([feature.flatten(1) for feature in features], dim=1)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        features = self._extract_features(x)
        inputs: List[torch.Tensor] = [features] * (self.num_targets + 1)
        for block in self.blocks:
            inputs = block(inputs)
        tower_features = [tower(inputs[target_idx]) for target_idx, tower in enumerate(self.towers)]
        target_logits = torch.stack([head(tower_features[target_idx]) for target_idx, head in enumerate(self.heads)], dim=1)
        logits = target_logits.mean(dim=1) if self.output_mode == 'mean' else target_logits[:, self.output_index]
        if returnt in ('out', 'logits'):
            return logits
        if returnt in ('both', 'full', 'all'):
            return logits, target_logits
        if returnt == 'features':
            return torch.cat(tower_features, dim=1)
        raise ValueError(f'Unsupported returnt value for MultiAngentBackbone: {returnt}')


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


class DerppMultiAngent(ContinualModel):
    """DER++ with independent MultiAngent and multi-backbone ablation switches."""

    NAME = 'derpp-multiangent'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight for DER++ logit replay.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight for DER++ label replay.')
        parser.add_argument('--use_multi_backbone', type=int, default=0,
                            help='Use all --multi_backbones when set to 1. If 0, only the primary --backbone is used.')
        parser.add_argument('--multi_backbones', type=str, default=None,
                            help='Comma-separated backbone names for multi-backbone ablations, e.g. `vit,clip`.')
        parser.add_argument('--use_multiangent', type=int, default=0,
                            help='Use MultiAngent feature-level expert fusion when set to 1. If 0, use DER++ logits or logits fusion.')
        parser.add_argument('--fusion_mode', type=str, default='linear_attention', choices=['linear_attention', 'manual'],
                            help='Logits-level multi-backbone fusion used when --use_multiangent 0 and --use_multi_backbone 1.')
        parser.add_argument('--fusion_weights', type=str, default=None,
                            help='Comma-separated backbone weights for --fusion_mode manual, e.g. `0.7,0.3`.')
        parser.add_argument('--multiangent_num_targets', type=int, default=2, help='Number of MultiAngent target branches.')
        parser.add_argument('--multiangent_num_levels', type=int, default=2, help='Number of stacked MultiAngent extraction levels.')
        parser.add_argument('--multiangent_shared_expert_num', type=int, default=1, help='Number of shared experts in each MultiAngent level.')
        parser.add_argument('--multiangent_specific_expert_num', type=int, default=1, help='Number of target-specific experts per target in each level.')
        parser.add_argument('--multiangent_expert_dim', type=int, default=128, help='Output dimension of each MultiAngent expert and tower.')
        parser.add_argument('--multiangent_expert_hidden_units', type=str, default='256', help='Comma-separated hidden units for each expert MLP.')
        parser.add_argument('--multiangent_gate_hidden_units', type=str, default='', help='Comma-separated hidden units for each gate MLP.')
        parser.add_argument('--multiangent_tower_hidden_units', type=str, default='64', help='Comma-separated hidden units for each tower MLP.')
        parser.add_argument('--multiangent_dropout', type=float, default=0.0, help='Dropout probability inside MultiAngent modules.')
        parser.add_argument('--multiangent_activation', type=str, default='relu', choices=['relu', 'gelu', 'tanh'], help='Activation used inside MultiAngent modules.')
        parser.add_argument('--multiangent_output_mode', type=str, default='mean', choices=['mean', 'target'], help='Average target logits or select one target branch.')
        parser.add_argument('--multiangent_output_index', type=int, default=0, help='Target branch index used when --multiangent_output_mode target.')
        parser.add_argument('--clip_model_name', type=str, default='ViT-B-16', help='CLIP architecture name used when `clip` is listed in --multi_backbones.')
        parser.add_argument('--clip_checkpoint_path', type=str, default=None, help='Local CLIP checkpoint path used when `clip` is listed in --multi_backbones.')
        parser.add_argument('--freeze_clip', type=int, default=0, help='Freeze the CLIP visual encoder when `clip` is listed in --multi_backbones.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        num_classes = dataset.N_CLASSES if dataset is not None else args.n_classes
        backbones = self._build_backbones(backbone, args, num_classes)
        if int(args.use_multiangent):
            net = MultiAngentBackbone(
                backbones, num_classes, args.multiangent_num_targets, args.multiangent_num_levels,
                args.multiangent_shared_expert_num, args.multiangent_specific_expert_num, args.multiangent_expert_dim,
                args.multiangent_expert_hidden_units, args.multiangent_gate_hidden_units, args.multiangent_tower_hidden_units,
                args.multiangent_dropout, args.multiangent_activation, args.multiangent_output_mode, args.multiangent_output_index)
        elif len(backbones) > 1:
            net = LogitFusionBackbone(backbones, num_classes, fusion_mode=args.fusion_mode, fusion_weights=args.fusion_weights)
        else:
            net = backbones[0]
        super().__init__(net, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def _build_backbones(self, backbone, args, num_classes: int) -> List[nn.Module]:
        primary_name = _normalize_backbone_name(args.backbone)
        names = _parse_backbone_names(args.multi_backbones, args.backbone) if int(args.use_multi_backbone) else [primary_name]
        backbones = []
        used_primary = False
        for name in names:
            if name == primary_name and not used_primary:
                backbones.append(backbone)
                used_primary = True
            else:
                logging.info('Building auxiliary backbone `%s` for %s.', name, self.NAME)
                backbones.append(_build_registered_backbone(name, copy.copy(args), num_classes))
        if not used_primary:
            logging.warning('Primary --backbone `%s` is not listed in --multi_backbones and will not be used.', args.backbone)
        return backbones

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

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item()
