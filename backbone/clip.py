"""Local-checkpoint CLIP image backbone."""

import os

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _resolve_clip_checkpoint(clip_checkpoint_path: str = None) -> str:
    candidates = [clip_checkpoint_path, os.environ.get('MAMMOTH_CLIP_PRETRAINED_PATH')]
    candidates.extend([
        os.path.join('checkpoints', 'clip', 'ViT-B-16.pt'),
        os.path.join('checkpoints', 'ViT-B-16.pt'),
    ])
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    return os.path.expanduser(clip_checkpoint_path) if clip_checkpoint_path else None


class LocalCLIPBackbone(MammothBackbone):
    """CLIP visual encoder plus a trainable classifier head loaded only from local files."""

    def __init__(self, num_classes: int, clip_model_name: str = 'ViT-B-16', clip_checkpoint_path: str = None, freeze_clip: int = 0) -> None:
        super().__init__()
        checkpoint_path = _resolve_clip_checkpoint(clip_checkpoint_path)
        if checkpoint_path is None:
            raise FileNotFoundError(
                'A local CLIP checkpoint is required. Set --clip_checkpoint_path, '
                'MAMMOTH_CLIP_PRETRAINED_PATH, or place ViT-B-16.pt under ./checkpoints/clip/.'
            )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Local CLIP checkpoint not found: {checkpoint_path}')

        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=checkpoint_path)
        self.visual = clip_model.visual
        feature_dim = getattr(self.visual, 'output_dim', None)
        if feature_dim is None:
            feature_dim = getattr(clip_model, 'embed_dim', None)
        if feature_dim is None and hasattr(clip_model, 'text_projection'):
            feature_dim = clip_model.text_projection.shape[1]
        if feature_dim is None:
            raise ValueError(f'Could not infer CLIP visual feature dimension for `{clip_model_name}`.')
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('clip_mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('clip_std', torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

        if freeze_clip:
            for param in self.visual.parameters():
                param.requires_grad = False

    def _to_clip_normalization(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.imagenet_std.to(dtype=x.dtype) + self.imagenet_mean.to(dtype=x.dtype)
        return (x - self.clip_mean.to(dtype=x.dtype)) / self.clip_std.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, returnt: str = 'out'):
        clip_x = self._to_clip_normalization(x)
        features = self.visual(clip_x)
        if features.dtype != self.classifier.weight.dtype:
            features = features.to(self.classifier.weight.dtype)
        logits = self.classifier(features)
        if returnt == 'features':
            return features
        if returnt in ('both', 'full', 'all'):
            return logits, features
        if returnt in ('out', 'logits'):
            return logits
        raise ValueError(f'Unsupported returnt value for LocalCLIPBackbone: {returnt}')


@register_backbone('clip')
def clip_backbone(num_classes: int, clip_model_name: str = 'ViT-B-16', clip_checkpoint_path: str = None, freeze_clip: int = 0):
    return LocalCLIPBackbone(num_classes=num_classes, clip_model_name=clip_model_name, clip_checkpoint_path=clip_checkpoint_path, freeze_clip=freeze_clip)
