import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiDataset, SimpleAugment, get_dataset_configs
from loss import BinaryIoU, BinaryMaskLoss

SAM2_TYPES = ("tiny", "small", "base", "large")
SAM_TYPES = ("vit_b", "vit_l", "vit_h")
SAM_CKPT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_sam_ckpt(model_type, sam_ckpt=None, ckpt_dir=None):
    if sam_ckpt:
        return sam_ckpt
    if os.environ.get("SAM_CKPT"):
        return os.environ["SAM_CKPT"]
    ckpt_dir = ckpt_dir or os.environ.get("SAM_CKPT_DIR", "./checkpoints")
    return os.path.join(ckpt_dir, SAM_CKPT_FILENAMES[model_type])


def build_from_args(args, organ_num, pretrained=True):
    if args.arch == "sam2":
        from model import build_model, load_pretrained, resolve_ckpt
        model = build_model(
            model_type=args.model_type,
            organ_num=organ_num,
            img_size=args.img_size,
            common_ratio=args.common_ratio,
            expert_ratio=args.expert_ratio,
            prompt_threshold=args.prompt_threshold,
        )
        if pretrained:
            load_pretrained(model, resolve_ckpt(args.model_type, args.sam2_ckpt))
        return model

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "未找到 segment_anything: 使用 --arch sam 需要先安装 SAM1 分支, 例如\n"
            "    pip install -e /path/to/segment-anything-main\n"
            "或把该目录加入 PYTHONPATH。"
        ) from exc

    model = sam_model_registry[f"memo_{args.model_type}"](
        organ_num=organ_num,
        checkpoint=resolve_sam_ckpt(args.model_type, args.sam_ckpt) if pretrained else None,
        img_size=args.img_size,
        common_ratio=args.common_ratio,
        expert_ratio=args.expert_ratio,
        prompt_threshold=args.prompt_threshold,
    )
    return model


def train_one_fold(args, fold, dataset_configs):
    print(f"\n{'='*20} Starting Training: Fold {fold} Model: {args.model_type} {'='*20}")

    fold_seed = args.seed + fold
    set_seed(fold_seed)
    print(f"Seed: {fold_seed}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")

    save_dir = os.path.join(args.save_dir, f"{args.model_type}_fold_{fold}")
    os.makedirs(save_dir, exist_ok=True)

    train_ds = MultiDataset(dataset_configs, fold=fold, phase="train",
                            img_size=args.img_size, transform=SimpleAugment(),
                            seed=args.seed)
    val_ds = MultiDataset(dataset_configs, fold=fold, phase="val",
                          img_size=args.img_size, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              generator=torch.Generator().manual_seed(fold_seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn)

    organ_num = len(dataset_configs)
    model = build_from_args(args, organ_num, pretrained=True)

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = ("p_organ_all" in name or "experts" in name)
        else:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} / "
          f"{sum(p.numel() for p in model.parameters())}")

    model = model.to(device)

    criterion = BinaryMaskLoss().to(device)

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_dice = [], []

        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            organ_ids = batch["label"].to(device)

            optimizer.zero_grad()
            pred_masks, coarse_masks, _ = model(images, organ_ids)

            pred_masks = torch.sigmoid(pred_masks)
            coarse_masks = torch.sigmoid(coarse_masks)
            loss = criterion(pred_masks, masks) + criterion(coarse_masks, masks)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            with torch.no_grad():
                pred_binary = (pred_masks > 0.5).float()
                intersection = pred_binary * masks
                dice = (2. * intersection.sum()) / (pred_binary.sum() + masks.sum() + 1e-6)
                train_dice.append(dice.item())

            pbar.set_postfix({"loss": f"{np.mean(train_loss):.4f}",
                              "dice": f"{np.mean(train_dice):.4f}"})

        scheduler.step()

        model.eval()
        val_dice, val_iou = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.epochs} [Val]"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                organ_ids = batch["label"].to(device)

                pred_masks, _, _ = model(images, organ_ids)
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()

                intersection = (pred_binary * masks).sum(dim=(1, 2, 3))
                pred_sum = pred_binary.sum(dim=(1, 2, 3))
                mask_sum = masks.sum(dim=(1, 2, 3))
                val_dice.extend(((2. * intersection + 1e-6) / (pred_sum + mask_sum + 1e-6)).cpu().numpy())
                val_iou.extend(((intersection + 1e-6) / (pred_sum + mask_sum - intersection + 1e-6)).cpu().numpy())

        avg_val_dice, avg_val_iou = np.mean(val_dice), np.mean(val_iou)
        print(f"Fold {fold} Epoch {epoch+1} Summary: Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model for {args.model_type} Fold {fold} saved at {save_path} "
                  f"with Dice: {best_val_dice:.4f}")

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Continuous 5-fold Training for MEMO-USeg (SAM2 / SAM1)")
    parser.add_argument("--arch", type=str, default="sam2", choices=["sam2", "sam"],
                        help="骨干后端: sam2 (Hiera tiny/small/base/large) 或 sam (ViT vit_b/vit_l/vit_h)")
    parser.add_argument("--model_type", type=str, default=None,
                        help="SAM2: tiny/small/base/large; SAM1: vit_b/vit_l/vit_h")
    parser.add_argument("--data_config", type=str, default=None,
                        help="数据集 YAML 配置, 默认读取环境变量 DATASETS_CONFIG 或 ./datasets.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default=None,
                        help="SAM2.1 预训练权重; 默认按 SAM2_CKPT_DIR/<对应尺寸>.pt 查找")
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="SAM1 预训练权重; 默认按 SAM_CKPT_DIR/<对应尺寸>.pth 查找")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="默认 SAM2: work_dir/MEMO_SAM2_5Fold, SAM1: work_dir/MEMO_SAM_5Fold")
    parser.add_argument("--common_ratio", type=float, default=4.0,
                        help="通用专家扩张比 R_GE (论文 4.0)")
    parser.add_argument("--expert_ratio", type=float, default=0.25,
                        help="特定专家扩张比 R_SE (论文 0.25)")
    parser.add_argument("--prompt_threshold", type=float, default=0.9,
                        help="SPG 阈值 tau (论文 0.9)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="随机种子; 第 fold 折使用 seed+fold")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="指定折号, 如 --folds 0 2 4; 优先于 start/end_fold")
    args = parser.parse_args()

    allowed = SAM2_TYPES if args.arch == "sam2" else SAM_TYPES
    if args.model_type is None:
        args.model_type = "large" if args.arch == "sam2" else "vit_l"
    if args.model_type not in allowed:
        parser.error(f"--arch {args.arch} 仅支持 {allowed}, 当前为 {args.model_type}")
    if args.save_dir is None:
        args.save_dir = ("work_dir/MEMO_SAM2_5Fold" if args.arch == "sam2"
                         else "work_dir/MEMO_SAM_5Fold")

    configs = get_dataset_configs(args.data_config)
    folds_to_run = args.folds if args.folds is not None else range(args.start_fold, args.end_fold)

    fold_results = []
    for fold in folds_to_run:
        fold_results.append(train_one_fold(args, fold, configs))

    print("\n" + "=" * 50)
    print("5-Fold Training Completed!")
    for i, res in enumerate(fold_results):
        print(f"Fold {folds_to_run[i]}: Best Val Dice = {res:.4f}")
    if fold_results:
        print(f"Average Best Dice: {np.mean(fold_results):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
