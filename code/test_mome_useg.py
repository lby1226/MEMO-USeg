import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiDataset, get_dataset_configs
from metrics import calculate_metrics

SAM2_TYPES = ("tiny", "small", "base", "large")
SAM_TYPES = ("vit_b", "vit_l", "vit_h")
SAM_CKPT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def resolve_sam_ckpt(model_type, sam_ckpt=None, ckpt_dir=None):
    if sam_ckpt:
        return sam_ckpt
    if os.environ.get("SAM_CKPT"):
        return os.environ["SAM_CKPT"]
    ckpt_dir = ckpt_dir or os.environ.get("SAM_CKPT_DIR", "./checkpoints")
    return os.path.join(ckpt_dir, SAM_CKPT_FILENAMES[model_type])


def build_from_args(args, organ_num, pretrained=False):
    if args.arch == "sam2":
        from model import build_model
        return build_model(
            model_type=args.model_type,
            organ_num=organ_num,
            img_size=args.img_size,
            common_ratio=args.common_ratio,
            expert_ratio=args.expert_ratio,
            prompt_threshold=args.prompt_threshold,
        )

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "未找到 segment_anything: 使用 --arch sam 需要先安装 SAM1 分支, 例如\n"
            "    pip install -e /path/to/segment-anything-main\n"
            "或把该目录加入 PYTHONPATH。"
        ) from exc

    return sam_model_registry[f"memo_{args.model_type}"](
        organ_num=organ_num,
        checkpoint=resolve_sam_ckpt(args.model_type, args.sam_ckpt) if pretrained else None,
        img_size=args.img_size,
        common_ratio=args.common_ratio,
        expert_ratio=args.expert_ratio,
        prompt_threshold=args.prompt_threshold,
    )


def remove_small_objects(mask, min_area=100):
    if not np.any(mask):
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[labels == i] = 1
    return new_mask.astype(np.uint8)


def visualize_error_analysis(orig_img, orig_mask, pred_prob, alpha=0.7):
    h, w = orig_img.shape[:2]
    pred_prob_up = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred_prob_up > 0.5).astype(np.uint8)
    pred_mask = remove_small_objects(pred_mask, min_area=100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    if orig_mask.max() > 1:
        orig_mask_bin = (orig_mask > 127).astype(np.uint8)
    else:
        orig_mask_bin = (orig_mask > 0.5).astype(np.uint8)
    orig_mask = cv2.resize(orig_mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    overlap = orig_img.copy().astype(np.float32)
    tp = (pred_mask == 1) & (orig_mask == 1)
    fp = (pred_mask == 1) & (orig_mask == 0)
    fn = (pred_mask == 0) & (orig_mask == 1)
    overlap[tp] = overlap[tp] * (1 - alpha) + np.array([0, 180, 0]) * alpha
    overlap[fp] = overlap[fp] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    overlap[fn] = overlap[fn] * (1 - alpha) + np.array([255, 120, 0]) * alpha
    return np.clip(overlap, 0, 255).astype(np.uint8)


def save_visualizations(image_path, mask_path, pred_prob, save_dir, dataset_name, fold_idx):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        return
    orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig_mask is None:
        orig_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    vis = visualize_error_analysis(orig_img, orig_mask, pred_prob, alpha=0.7)
    out_dir = os.path.join(save_dir, f"fold_{fold_idx}_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{dataset_name}_{basename}.png"), vis)


def test_fold(model, dataloader, device, args, fold_idx):
    model.eval()
    dataset_results, sample_results = {}, []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing Fold {fold_idx}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            organ_ids = batch["label"].to(device)

            pred_masks, _, _ = model(images, organ_ids)
            if pred_masks.shape[2:] != masks.shape[2:]:
                pred_masks = F.interpolate(pred_masks, size=masks.shape[2:],
                                           mode="bilinear", align_corners=False)
            probas = torch.sigmoid(pred_masks)
            preds = (probas > 0.5).squeeze(1).cpu().numpy()
            dist_size = 224 if args.img_size == 1024 else None
            batch_metrics = calculate_metrics(preds, masks, probas, dist_size=dist_size)

            for i in range(images.shape[0]):
                name = batch["dataset_name"][i]
                if name not in dataset_results:
                    dataset_results[name] = {k: [] for k in batch_metrics}
                sample_info = {"dataset": name, "filename": batch["filename"][i], "fold": fold_idx}
                for k, v in batch_metrics.items():
                    dataset_results[name][k].append(v[i])
                    sample_info[k] = v[i]
                sample_results.append(sample_info)
                if (args.visualize and args.visualize_only in (None, name)
                        and args.visualize_fold in (None, fold_idx)):
                    save_visualizations(batch["path"][i], batch["mask_path"][i],
                                        probas[i, 0].cpu().numpy(),
                                        args.save_path, name, fold_idx)

    final_dataset_metrics = {name: {k: np.mean(v) for k, v in m.items()}
                             for name, m in dataset_results.items()}
    macro = {k: np.mean([m[k] for m in final_dataset_metrics.values()])
             for k in batch_metrics}
    return macro, final_dataset_metrics, sample_results


def main():
    parser = argparse.ArgumentParser("MEMO-USeg 5-fold testing (SAM2 / SAM1)")
    parser.add_argument("--arch", type=str, default="sam2", choices=["sam2", "sam"],
                        help="骨干后端: sam2 (Hiera tiny/small/base/large) 或 sam (ViT vit_b/vit_l/vit_h)")
    parser.add_argument("--model_type", type=str, default=None,
                        help="SAM2: tiny/small/base/large; SAM1: vit_b/vit_l/vit_h")
    parser.add_argument("--data_config", type=str, default=None,
                        help="数据集 YAML 配置, 默认读取环境变量 DATASETS_CONFIG 或 ./datasets.yaml")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="默认 SAM2: work_dir/MEMO_SAM2_5Fold, SAM1: work_dir/MEMO_SAM_5Fold")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="SAM1 预训练权重; 仅在需要构建带预训练初始化时使用")
    parser.add_argument("--prompt_threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234,
                        help="随机种子, 需与 train.py 一致以保证折划分相同")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_only", type=str, default=None, help="仅可视化指定数据集")
    parser.add_argument("--visualize_fold", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--common_ratio", type=float, default=4.0)
    parser.add_argument("--expert_ratio", type=float, default=0.25)
    args = parser.parse_args()

    allowed = SAM2_TYPES if args.arch == "sam2" else SAM_TYPES
    if args.model_type is None:
        args.model_type = "large" if args.arch == "sam2" else "vit_l"
    if args.model_type not in allowed:
        parser.error(f"--arch {args.arch} 仅支持 {allowed}, 当前为 {args.model_type}")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = ("work_dir/MEMO_SAM2_5Fold" if args.arch == "sam2"
                               else "work_dir/MEMO_SAM_5Fold")
    if args.save_path is None:
        tag = "sam2" if args.arch == "sam2" else "sam"
        args.save_path = f"./test_results_{tag}_{args.model_type}"
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    configs = get_dataset_configs(args.data_config)
    organ_num = len(configs)

    all_fold_macro, all_fold_dataset, all_samples = [], {}, []
    for fold in args.folds:
        test_ds = MultiDataset(configs, fold=fold, phase="test",
                               img_size=args.img_size, seed=args.seed)
        if not test_ds:
            continue
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = build_from_args(args, organ_num, pretrained=False).to(device)

        ckpt = os.path.join(args.checkpoint_dir, f"{args.model_type}_fold_{fold}", "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"Skipping fold {fold}: {ckpt} not found")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        print(f"Loaded: {ckpt}")

        macro, per_dataset, samples = test_fold(model, test_loader, device, args, fold)
        all_fold_macro.append(macro)
        all_samples.extend(samples)
        for name, m in per_dataset.items():
            if name not in all_fold_dataset:
                all_fold_dataset[name] = {k: [] for k in m}
            for k, v in m.items():
                all_fold_dataset[name][k].append(v)

    if not all_fold_macro:
        return

    pd.DataFrame(all_samples).to_csv(
        os.path.join(args.save_path, "sample_fold_means_for_significance.csv"), index=False)

    summary_data = []
    for name, m in all_fold_dataset.items():
        row = {"Dataset": name}
        for k in m:
            row[f"{k}_mean"] = np.mean(m[k])
            row[f"{k}_std"] = np.std(m[k], ddof=1)
        summary_data.append(row)

    overall_row = {"Dataset": "OVERALL_MACRO"}
    for k in all_fold_macro[0]:
        vals = [m[k] for m in all_fold_macro]
        overall_row[f"{k}_mean"] = np.mean(vals)
        overall_row[f"{k}_std"] = np.std(vals, ddof=1)
    summary_data.append(overall_row)

    pd.DataFrame(summary_data).to_csv(
        os.path.join(args.save_path, "final_statistics.csv"), index=False)

    excel_data = []
    for r in summary_data:
        ex_row = {"Dataset": r["Dataset"]}
        for k in ["dice", "iou", "sm"]:
            ex_row[k] = f"{r[f'{k}_mean']*100:.2f}±{r[f'{k}_std']*100:.2f}"
        for k in ["hd95", "assd"]:
            ex_row[k] = f"{r[f'{k}_mean']:.2f}±{r[f'{k}_std']:.2f}"
        excel_data.append(ex_row)
    pd.DataFrame(excel_data).to_excel(
        os.path.join(args.save_path, "final_statistics.xlsx"), index=False)
    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
