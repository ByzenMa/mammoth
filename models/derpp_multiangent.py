"""DER++ with independently configurable MultiAngent and MINE-loss ablations."""

import math
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


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


class MINE(nn.Module):
    """Mutual Information Neural Estimator used as an optional auxiliary loss."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.LazyLinear(hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_marginal = torch.roll(y, shifts=1, dims=0)
        joint_logits = self.layers(torch.cat([x, y], dim=1))
        marginal_logits = self.layers(torch.cat([x, y_marginal], dim=1))
        log_mean_exp = torch.logsumexp(marginal_logits, dim=0) - math.log(marginal_logits.size(0))
        mi_estimate = math.log2(math.e) * (torch.mean(joint_logits) - log_mean_exp.squeeze(0))
        return F.softplus(-mi_estimate)


class MINELossModule(nn.Module):
    """Computes pairwise MINE loss over projected chunks of one backbone feature vector."""

    def __init__(self, num_views: int, projection_dim: int, hidden_size: int) -> None:
        super().__init__()
        if num_views < 2:
            raise ValueError('--mine_num_views must be at least 2.')
        self.num_views = num_views
        self.projections = nn.ModuleList([nn.LazyLinear(projection_dim) for _ in range(num_views)])
        self.estimators = nn.ModuleList([MINE(hidden_size) for _ in range(num_views * (num_views - 1) // 2)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(features.flatten(1), self.num_views, dim=1)
        views = [projection(chunk) for projection, chunk in zip(self.projections, chunks)]
        losses = []
        estimator_idx = 0
        for left_idx in range(self.num_views):
            for right_idx in range(left_idx + 1, self.num_views):
                losses.append(self.estimators[estimator_idx](views[left_idx], views[right_idx]))
                estimator_idx += 1
        return torch.stack(losses).mean()


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
        specific_outputs = [[expert(inputs[target_idx]) for expert in self.specific_experts[target_idx]] for target_idx in range(self.num_targets)]
        shared_outputs = [expert(inputs[-1]) for expert in self.shared_experts]
        outputs = [self._gate_experts(specific_outputs[target_idx] + shared_outputs, self.target_gates[target_idx](inputs[target_idx])) for target_idx in range(self.num_targets)]
        if not self.is_last:
            shared_experts = [expert for target_experts in specific_outputs for expert in target_experts] + shared_outputs
            outputs.append(self._gate_experts(shared_experts, self.shared_gate(inputs[-1])))
        return outputs


class DerppMultiAngentNet(nn.Module):
    """Single-backbone classifier with optional MultiAngent head and optional MINE loss."""

    def __init__(self, backbone: nn.Module, num_classes: int, use_multiangent: bool,
                 num_targets: int, num_levels: int, shared_expert_num: int, specific_expert_num: int, expert_dim: int,
                 expert_hidden_units: str, gate_hidden_units: str, tower_hidden_units: str, dropout: float,
                 activation: str, output_mode: str, output_index: int, use_mine_loss: bool,
                 mine_num_views: int, mine_projection_dim: int, mine_hidden_size: int) -> None:
        super().__init__()
        if num_targets < 1:
            raise ValueError('--multiangent_num_targets must be at least 1.')
        if num_levels < 1:
            raise ValueError('--multiangent_num_levels must be at least 1.')
        if output_mode not in ['mean', 'target']:
            raise ValueError('--multiangent_output_mode must be `mean` or `target`.')
        if output_mode == 'target' and not 0 <= output_index < num_targets:
            raise ValueError(f'--multiangent_output_index must be in [0, {num_targets - 1}] for {num_targets} targets.')
        self.backbone = backbone
        self.use_multiangent = use_multiangent
        self.output_mode = output_mode
        self.output_index = output_index
        self.last_mine_loss = None
        self.mine = MINELossModule(mine_num_views, mine_projection_dim, mine_hidden_size) if use_mine_loss else None
        if use_multiangent:
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
        return self.backbone(x, returnt='features').flatten(1)

    def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
        if not self.use_multiangent and self.mine is None:
            return self.backbone(x, returnt=returnt)
        features = self._extract_features(x)
        self.last_mine_loss = self.mine(features) if self.mine is not None else None
        if not self.use_multiangent:
            logits = self.backbone(x)
            aux_features = features
        else:
            inputs: List[torch.Tensor] = [features] * (len(self.towers) + 1)
            for block in self.blocks:
                inputs = block(inputs)
            tower_features = [tower(inputs[target_idx]) for target_idx, tower in enumerate(self.towers)]
            target_logits = torch.stack([head(tower_features[target_idx]) for target_idx, head in enumerate(self.heads)], dim=1)
            logits = target_logits.mean(dim=1) if self.output_mode == 'mean' else target_logits[:, self.output_index]
            aux_features = torch.cat(tower_features, dim=1)
        if returnt in ('out', 'logits'):
            return logits
        if returnt in ('both', 'full', 'all'):
            return logits, aux_features
        if returnt == 'features':
            return aux_features
        raise ValueError(f'Unsupported returnt value for DerppMultiAngentNet: {returnt}')

    def get_mine_loss(self) -> torch.Tensor:
        if self.last_mine_loss is None:
            raise RuntimeError('MINE loss requested before a forward pass with --use_mine_loss 1.')
        return self.last_mine_loss


class DerppMultiAngent(ContinualModel):
    """DER++ with independent MultiAngent and MINE-loss ablation switches."""

    NAME = 'derpp-multiangent'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight for DER++ logit replay.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight for DER++ label replay.')
        parser.add_argument('--use_multiangent', type=int, default=0, help='Use MultiAngent feature-level expert head when set to 1.')
        parser.add_argument('--use_mine_loss', type=int, default=0, help='Add the MINE auxiliary loss when set to 1.')
        parser.add_argument('--mine_loss_weight', type=float, default=0.1, help='Weight of the optional MINE auxiliary loss.')
        parser.add_argument('--mine_num_views', type=int, default=2, help='Number of feature chunks used to compute pairwise MINE loss.')
        parser.add_argument('--mine_projection_dim', type=int, default=128, help='Projection dimension for each MINE feature view.')
        parser.add_argument('--mine_hidden_size', type=int, default=64, help='Hidden size of each MINE estimator.')
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
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.mine_loss_weight < 0:
            raise ValueError('--mine_loss_weight must be non-negative so the total loss cannot be reduced by the auxiliary MINE term.')
        num_classes = dataset.N_CLASSES if dataset is not None else args.n_classes
        net = DerppMultiAngentNet(
            backbone, num_classes, bool(args.use_multiangent), args.multiangent_num_targets, args.multiangent_num_levels,
            args.multiangent_shared_expert_num, args.multiangent_specific_expert_num, args.multiangent_expert_dim,
            args.multiangent_expert_hidden_units, args.multiangent_gate_hidden_units, args.multiangent_tower_hidden_units,
            args.multiangent_dropout, args.multiangent_activation, args.multiangent_output_mode, args.multiangent_output_index,
            bool(args.use_mine_loss), args.mine_num_views, args.mine_projection_dim, args.mine_hidden_size)
        super().__init__(net, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def _mine_loss(self) -> torch.Tensor:
        return self.net.get_mine_loss() if int(self.args.use_mine_loss) else 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if int(self.args.use_mine_loss):
            loss += self.args.mine_loss_weight * self._mine_loss()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            if int(self.args.use_mine_loss):
                loss += self.args.mine_loss_weight * self._mine_loss()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            if int(self.args.use_mine_loss):
                loss += self.args.mine_loss_weight * self._mine_loss()

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item()
