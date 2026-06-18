"""Domain-incremental pneumonia X-ray dataset.

This dataset treats three chest X-ray corpora as three domains with a shared
binary label space: ``normal`` (0) and ``pneumonia / lung opacity`` (1).
"""

import csv
import logging
import os
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from utils.conf import base_path
from utils.prompt_templates import templates


CHEST_XRAY_SIZE = (224, 224)


class BinaryChestDomainDataset(Dataset):
    """Simple image dataset returning Mammoth's train tuple."""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform: Optional[Callable] = None, train: bool = True, size: Tuple[int, int] = CHEST_XRAY_SIZE) -> None:
        self.samples = list(samples)
        self.data = np.array([path for path, _ in self.samples])
        self.targets = np.array([target for _, target in self.samples], dtype=np.int64)
        self.transform = transform
        self.train = train
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(size=size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        if path.lower().endswith('.dcm'):
            try:
                import pydicom
            except ImportError as e:
                raise ImportError('Reading RSNA DICOM files requires the optional dependency `pydicom`. Install it with `pip install pydicom`.') from e

            dicom = pydicom.dcmread(path)
            arr = dicom.pixel_array.astype(np.float32)
            arr -= arr.min()
            max_value = arr.max()
            if max_value > 0:
                arr /= max_value
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr, mode='L').convert('RGB')

        return Image.open(path).convert('RGB')

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self._load_image(path)
        not_aug_img = self.not_aug_transform(img.copy())
        if self.transform is not None:
            img = self.transform(img)
        if not self.train:
            return img, target
        return img, target, not_aug_img


class DomainPneumonia(ContinualDataset):
    """Three-domain binary pneumonia dataset.

    Domains are loaded in this order:
    1. ``chest_xray`` folder dataset (NORMAL vs PNEUMONIA).
    2. ``CheXpert-v1.0-small`` filtered to normal vs pneumonia-related findings.
    3. ``rsna-pneumonia-detection-challenge`` filtered to Normal vs Lung Opacity.
    """

    NAME = 'domain-pneumonia'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 2
    N_CLASSES = 2
    N_TASKS = 3
    SIZE = CHEST_XRAY_SIZE
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    LABELS = ['normal', 'pneumonia']
    DOMAIN_NAMES = ['chest_xray', 'chexpert_small', 'rsna_pneumonia']

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    TRANSFORM = transforms.Compose([
        transforms.Resize(size=SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    TEST_TRANSFORM = TRANSFORM

    def __init__(self, args, medical_domain_root: str = None, medical_domain_val_ratio: float = 0.2) -> None:
        super().__init__(args)
        self.medical_domain_root = self._resolve_root(medical_domain_root)
        self.medical_domain_val_ratio = medical_domain_val_ratio
        self._domain_cache = None

    @staticmethod
    def _resolve_root(root: Optional[str]) -> str:
        candidates = []
        if root is not None:
            candidates.append(root)
        candidates.extend([
            os.path.join(base_path(), 'dataset'),
            os.path.join(base_path(), 'datasets'),
            os.path.join(os.getcwd(), 'dataset'),
            os.path.join(os.getcwd(), 'datasets'),
        ])
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate
        return candidates[0]

    def _split_samples(self, samples: Sequence[Tuple[str, int]], domain_name: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        samples = list(samples)
        if not samples:
            raise RuntimeError(f'No usable samples found for domain `{domain_name}` under `{self.medical_domain_root}`.')
        rng = np.random.RandomState(self.args.seed if self.args.seed is not None else 0)
        idxs = np.arange(len(samples))
        rng.shuffle(idxs)
        val_size = max(1, int(round(len(samples) * self.medical_domain_val_ratio))) if len(samples) > 1 else 1
        val_idxs = set(idxs[:val_size].tolist())
        train = [sample for i, sample in enumerate(samples) if i not in val_idxs]
        test = [sample for i, sample in enumerate(samples) if i in val_idxs]
        if not train:
            train, test = test, test
        return train, test

    def _load_chest_xray(self):
        root = os.path.join(self.medical_domain_root, 'chest_xray')
        samples = []
        for split in ['train', 'test', 'val']:
            for folder, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
                class_dir = os.path.join(root, split, folder)
                if not os.path.isdir(class_dir):
                    continue
                for dirpath, _, filenames in os.walk(class_dir):
                    for filename in filenames:
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            samples.append((os.path.join(dirpath, filename), label))
        return self._split_samples(samples, 'chest_xray')

    @staticmethod
    def _csv_float(row: dict, key: str) -> Optional[float]:
        value = row.get(key, '')
        if value == '' or value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _chexpert_label(self, row: dict) -> Optional[int]:
        relevant = ['Consolidation', 'Lung Opacity', 'Pneumonia']
        other_diseases = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Edema',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture'
        ]
        values = {key: self._csv_float(row, key) for key in ['No Finding', *relevant, *other_diseases]}
        if any(values[key] == -1.0 for key in relevant + other_diseases):
            return None
        if any(values[key] == 1.0 for key in other_diseases):
            return None
        if any(values[key] == 1.0 for key in relevant):
            return 1
        if values['No Finding'] == 1.0:
            return 0
        return None

    def _load_chexpert(self):
        root = os.path.join(self.medical_domain_root, 'CheXpert-v1.0-small')
        samples = []
        for csv_name in ['train.csv', 'valid.csv']:
            csv_path = os.path.join(root, csv_name)
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    label = self._chexpert_label(row)
                    if label is None:
                        continue
                    rel_path = row['Path']
                    if rel_path.startswith('CheXpert-v1.0-small/'):
                        rel_path = rel_path[len('CheXpert-v1.0-small/'):]
                    path = os.path.join(root, rel_path)
                    if os.path.isfile(path):
                        samples.append((path, label))
        return self._split_samples(samples, 'chexpert_small')

    def _load_rsna(self):
        root = os.path.join(self.medical_domain_root, 'rsna-pneumonia-detection-challenge')
        class_csv = os.path.join(root, 'stage_2_detailed_class_info.csv')
        image_dir = os.path.join(root, 'stage_2_train_images')
        patient_labels = OrderedDict()
        if not os.path.isfile(class_csv):
            raise FileNotFoundError(class_csv)
        with open(class_csv, newline='') as f:
            for row in csv.DictReader(f):
                patient_id = row['patientId']
                klass = row['class']
                if klass == 'Normal':
                    label = 0
                elif klass == 'Lung Opacity':
                    label = 1
                else:
                    continue
                patient_labels[patient_id] = max(label, patient_labels.get(patient_id, label))
        samples = []
        for patient_id, label in patient_labels.items():
            path = os.path.join(image_dir, f'{patient_id}.dcm')
            if os.path.isfile(path):
                samples.append((path, label))
        return self._split_samples(samples, 'rsna_pneumonia')

    def _load_domains(self):
        if self._domain_cache is None:
            self._domain_cache = [
                self._load_chest_xray(),
                self._load_chexpert(),
                self._load_rsna(),
            ]
            for domain_name, (train, test) in zip(self.DOMAIN_NAMES, self._domain_cache):
                logging.info('Loaded domain `%s`: %d train / %d test samples.', domain_name, len(train), len(test))
        return self._domain_cache

    def get_data_loaders(self):
        domain_idx = len(self.test_loaders)
        if domain_idx >= self.N_TASKS:
            domain_idx = self.N_TASKS - 1
        train_samples, test_samples = self._load_domains()[domain_idx]
        train_dataset = BinaryChestDomainDataset(train_samples, transform=self.TRANSFORM, train=True)
        test_dataset = BinaryChestDomainDataset(test_samples, transform=self.TEST_TRANSFORM, train=False)
        return store_masked_loaders(train_dataset, test_dataset, self)

    def get_class_names(self):
        if self.class_names is None:
            self.class_names = self.LABELS
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), DomainPneumonia.TRANSFORM])

    @set_default_from_args('backbone')
    def get_backbone():
        return 'vit'

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=DomainPneumonia.MEAN, std=DomainPneumonia.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(mean=DomainPneumonia.MEAN, std=DomainPneumonia.STD)

    @set_default_from_args('n_epochs')
    def get_epochs():
        return 10

    @set_default_from_args('batch_size')
    def get_batch_size():
        return 32
