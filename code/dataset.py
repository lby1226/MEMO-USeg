import json
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
IMAGE_DIR_NAMES = ('images', 'image', 'imgs', 'img')
MASK_DIR_NAMES = ('masks', 'mask', 'labels', 'label', 'annotations', 'annotations_png')
MASK_SUFFIXES = ('', '_mask', '_Mask', '-mask', '_1')

ORGAN_ORDER = (
    'TN3K', 'BUSI', 'fetal-head', 'kidneyUS', 'cdus', 'ovarian', 'fallmud',
    'fetal-lung', 'CC', 'Lymph Node', 'LUMINOUS', 'GIST514-DB', 'HMC-QU',
)


class SimpleAugment:
    def __init__(self, horizontal_flip=True, vertical_flip=True, random_rotate=True):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_rotate = random_rotate

    def __call__(self, image, mask):
        if self.horizontal_flip and random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if self.vertical_flip and random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if self.random_rotate and random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

        return {'image': image, 'mask': mask}


def _find_dir(parent, candidates):
    for name in candidates:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    return None


def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


SPLIT_DIR_NAMES = {
    'trainval': ('trainval', 'train', 'train-val'),
    'test': ('test', 'test-set', 'testset'),
}


def _scan_split(dataset_root, split):
    split_dir = _find_dir(dataset_root, SPLIT_DIR_NAMES[split])
    if split_dir is None:
        return []
    img_dir = _find_dir(split_dir, IMAGE_DIR_NAMES)
    mask_dir = _find_dir(split_dir, MASK_DIR_NAMES)
    if img_dir is None or mask_dir is None:
        print(f"[WARN] {split_dir} 缺少 images/ 或 masks/ 子目录, 已跳过")
        return []

    mask_index = {os.path.splitext(f)[0]: os.path.join(mask_dir, f)
                  for f in _list_images(mask_dir)}
    img_names = _list_images(img_dir)
    samples = []
    for img_name in img_names:
        stem = os.path.splitext(img_name)[0]
        mask_path = None
        for suffix in MASK_SUFFIXES:
            mask_path = mask_index.get(stem + suffix)
            if mask_path:
                break
        if mask_path is None:
            continue
        samples.append({'image': os.path.join(img_dir, img_name), 'mask': mask_path,
                        'stem': stem.lower()})
    if len(samples) < len(img_names):
        print(f"[WARN] {img_dir}: {len(img_names) - len(samples)} 张图未找到对应 mask")
    return samples


def _stem_of(item):
    if isinstance(item, dict):
        ref = item.get('image') or item.get('img') or item.get('name') or ''
    elif isinstance(item, str):
        ref = item
    elif isinstance(item, int):
        return f"{item:04d}"
    else:
        return None
    ref = str(ref).replace('\\', '/').rstrip('/')
    return os.path.splitext(os.path.basename(ref))[0].lower()


def _load_fold_json(dataset_root, fold):
    for pattern in (f'fold_{fold}.json', f'fold{fold}.json'):
        path = os.path.join(dataset_root, pattern)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def _split_trainval(samples, fold, fold_data, seed):
    total = len(samples)
    if total == 0:
        return [], []

    if fold_data is not None:
        if isinstance(fold_data, dict):
            val_items = fold_data.get('val') or fold_data.get('validation') or []
            train_items = fold_data.get('train')
        else:
            val_items, train_items = fold_data, None

        by_stem = {s['stem']: s for s in samples}
        val_stems = {_stem_of(i) for i in val_items} - {None}
        val = [by_stem[s] for s in val_stems if s in by_stem]
        if train_items is not None:
            train_stems = {_stem_of(i) for i in train_items} - {None}
            train = [by_stem[s] for s in train_stems if s in by_stem]
        else:
            train = [s for s in samples if s['stem'] not in val_stems]
        if val and train:
            return train, val
        print(f"[WARN] fold_{fold}.json 与实际文件不匹配 (val={len(val)}, train={len(train)}), "
              f"回退到自动划分")

    rng = np.random.RandomState(seed)
    order = rng.permutation(total)
    folds = np.array_split(order, 5)
    val_idx = set(folds[fold % 5].tolist())
    train = [samples[i] for i in range(total) if i not in val_idx]
    val = [samples[i] for i in val_idx]
    return train, val


def _canonical_class_ids(names):
    lower2name = {n.lower(): n for n in names}
    ordered = [n for n in ORGAN_ORDER if n.lower() in lower2name]
    extras = sorted(n for n in names if n.lower() not in {o.lower() for o in ORGAN_ORDER})
    return ordered + extras


def discover_dataset_configs(data_root):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据集根目录不存在: {data_root}")

    names = []
    for name in os.listdir(data_root):
        organ_root = os.path.join(data_root, name)
        if not os.path.isdir(organ_root) or name.startswith('.'):
            continue
        if _scan_split(organ_root, 'trainval') or _scan_split(organ_root, 'test'):
            names.append(name)
    if not names:
        raise FileNotFoundError(
            f"{data_root} 下未找到符合 <organ>/trainval/{{images,masks}} 的目录, "
            f"请检查目录结构")

    ordered = _canonical_class_ids(names)
    configs = []
    for class_id, name in enumerate(ordered):
        configs.append({
            'name': name,
            'root': os.path.join(data_root, name),
            'class_id': class_id,
            'mask_range': None,
        })
    return configs


def get_dataset_configs(config_path=None, data_root=None):
    if config_path is None:
        config_path = os.environ.get('DATASETS_CONFIG')
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        entries = data['datasets'] if isinstance(data, dict) else data
        configs = []
        for class_id, cfg in enumerate(entries):
            cfg = dict(cfg)
            cfg['root'] = os.path.expandvars(cfg['root'])
            cfg.setdefault('name', os.path.basename(os.path.normpath(cfg['root'])))
            cfg.setdefault('class_id', class_id)
            cfg.setdefault('mask_range', None)
            configs.append(cfg)
        return configs

    data_root = data_root or os.environ.get('DATASET_ROOT') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dataset')
    return discover_dataset_configs(data_root)


class MultiDataset(Dataset):
    def __init__(self, dataset_configs, fold=0, phase='train', transform=None,
                 img_size=224, seed=1234):
        assert phase in ('train', 'val', 'test'), f'未知 phase: {phase}'
        self.phase = phase
        self.fold = fold
        self.transform = transform
        self.img_size = img_size
        self.seed = seed
        self.all_data = []

        print(f"\n[{phase.upper()}] Dataset Loading Summary (Fold {fold}):")
        for config in dataset_configs:
            split = 'test' if phase == 'test' else 'trainval'
            samples = _scan_split(config['root'], split)
            if phase != 'test' and samples:
                fold_data = _load_fold_json(config['root'], fold)
                train, val = _split_trainval(samples, fold, fold_data, seed)
                samples = train if phase == 'train' else val

            for sample in samples:
                sample = dict(sample)
                sample['class_id'] = config['class_id']
                sample['dataset_name'] = config['name']
                sample['mask_range'] = config.get('mask_range')
                self.all_data.append(sample)
            print(f"  - {config['name']}: {len(samples)} samples")
        print(f"  - Total: {len(self.all_data)} samples\n")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]

        image = cv2.imread(data['image'])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {data['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(data['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩膜: {data['mask']}")
        mask_range = data['mask_range']
        if mask_range:
            low, high = mask_range
            mask = ((mask > low) & (mask < high)).astype(np.float32)
        elif mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()

        return {
            'image': image,
            'mask': mask,
            'label': data['class_id'],
            'dataset_name': data['dataset_name'],
            'filename': os.path.basename(data['image']),
            'path': data['image'],
            'mask_path': data['mask'],
        }


if __name__ == '__main__':
    configs = get_dataset_configs()
    print('Datasets (class_id: name):')
    for cfg in configs:
        print(f"  {cfg['class_id']:>2}: {cfg['name']}  ({cfg['root']})")

    for fold in range(5):
        train_ds = MultiDataset(configs, fold=fold, phase='train', seed=1234)
        val_ds = MultiDataset(configs, fold=fold, phase='val', seed=1234)
        overlap = {d['image'] for d in train_ds.all_data} & {d['image'] for d in val_ds.all_data}
        print(f"Fold {fold}: train={len(train_ds)} val={len(val_ds)} overlap={len(overlap)}")
    test_ds = MultiDataset(configs, fold=0, phase='test')
    print(f"Test: {len(test_ds)}")
