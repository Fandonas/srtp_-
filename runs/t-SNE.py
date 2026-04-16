#!/usr/bin/env python3

"""t-SNE visualization for CLIP-FSAR few-shot episodes."""

import argparse
import csv
import json
import os
import random
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.curdir))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

import utils.checkpoint as cu  # noqa: E402
from datasets.base.builder import build_loader  # noqa: E402
from models.base.builder import build_model  # noqa: E402
from utils.config import Config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize few-shot train/test embeddings with t-SNE."
    )
    parser.add_argument("--cfg", required=True, help="Path to the config yaml.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. If omitted, fallback to the project's checkpoint loading logic.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Few-shot split used to sample episodes.",
    )
    parser.add_argument(
        "--feature_modes",
        nargs="+",
        default=["target_raw", "target_ctx", "target_ctx_text"],
        choices=["target_raw", "target_ctx", "target_ctx_text"],
        help="Feature sets to export and plot.",
    )
    parser.add_argument(
        "--save_dir",
        default="output/tsne",
        help="Directory used to save embeddings and figures.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=200,
        help="Maximum number of few-shot episodes to sample.",
    )
    parser.add_argument(
        "--max_points_per_class",
        type=int,
        default=80,
        help="Maximum number of video points collected for each class and source.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="If > 0, randomly select this many classes first, then only collect samples from them.",
    )
    parser.add_argument(
        "--class_seed",
        type=int,
        default=None,
        help="Random seed used only for class subset selection. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--include_support",
        action="store_true",
        help="Also plot support samples in the same feature space.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It will be clipped to the valid range automatically.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=50,
        help="PCA dimension before t-SNE.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for this script.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--point_size",
        type=float,
        default=18.0,
        help="Scatter point size for video samples.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.72,
        help="Scatter alpha for video samples.",
    )

    args, cfg_opts = parser.parse_known_args()
    return args, cfg_opts


def load_cfg(args, cfg_opts):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--cfg", args.cfg] + cfg_opts
        cfg = Config(load=True)
    finally:
        sys.argv = old_argv
    return cfg


def prepare_cfg(cfg, args):
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device in the current project setup.")

    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = False
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = args.num_workers

    if args.checkpoint:
        cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_few_shot_head(model):
    net = unwrap_model(model)
    if not hasattr(net, "head"):
        raise RuntimeError("Top-level model does not expose a few-shot head.")
    head = net.head
    if not hasattr(head, "backbone") or not hasattr(head, "context2"):
        raise RuntimeError("Current head does not expose CLIP few-shot interfaces.")
    return head


def ensure_checkpoint_available(cfg):
    has_explicit = bool(getattr(cfg.TEST, "CHECKPOINT_FILE_PATH", ""))
    has_train_ckpt = bool(getattr(cfg.TRAIN, "CHECKPOINT_FILE_PATH", ""))
    has_output_ckpt = cu.has_checkpoint(cfg.OUTPUT_DIR)
    if not (has_explicit or has_train_ckpt or has_output_ckpt):
        raise RuntimeError(
            "No checkpoint was found. Please pass --checkpoint for the trained model you want to visualize."
        )


def move_episode_to_device(task_dict, device):
    episode = {}
    for key, value in task_dict.items():
        if torch.is_tensor(value):
            episode[key] = value[0].to(device, non_blocking=True)
        else:
            episode[key] = value
    return episode


def extract_raw_features(head, task_dict, use_amp):
    support_images = task_dict["support_set"]
    target_images = task_dict["target_set"]

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            support_features = head.backbone(support_images).squeeze()
            target_features = head.backbone(target_images).squeeze()

    dim = int(target_features.shape[1])
    support_features = support_features.reshape(-1, head.args.DATA.NUM_INPUT_FRAMES, dim)
    target_features = target_features.reshape(-1, head.args.DATA.NUM_INPUT_FRAMES, dim)
    return support_features.float(), target_features.float()


def enhance_with_context(head, raw_features, use_amp):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            enhanced = head.context2(raw_features, raw_features, raw_features)
    return enhanced.float()


def reduce_video_features(frame_features):
    pooled = frame_features.mean(dim=1)
    return F.normalize(pooled, dim=1)


def resolve_text_bank(head, split):
    if split == "train":
        if hasattr(head, "query_embed_train"):
            bank = head.query_embed_train.weight.detach()
        else:
            bank = head.text_features_train.detach()
        names = list(head.class_real_train)
    else:
        bank = head.text_features_test.detach()
        names = list(head.class_real_test)
    bank = F.normalize(bank.float(), dim=1)
    return bank, names


def select_class_subset(head, args):
    class_names = head.class_real_train if args.split == "train" else head.class_real_test
    all_class_ids = list(range(len(class_names)))

    if args.num_classes <= 0:
        return None
    if args.num_classes > len(all_class_ids):
        raise ValueError(
            f"--num_classes={args.num_classes} exceeds available classes ({len(all_class_ids)}) for split={args.split}."
        )

    class_rng = random.Random(args.class_seed if args.class_seed is not None else args.seed)
    selected_ids = sorted(class_rng.sample(all_class_ids, args.num_classes))
    return set(selected_ids)


def is_collection_complete(counts, selected_class_ids, include_support, max_points_per_class):
    if not selected_class_ids:
        return False

    for label_id in selected_class_ids:
        if counts.get(("target", int(label_id)), 0) < max_points_per_class:
            return False
        if include_support and counts.get(("support", int(label_id)), 0) < max_points_per_class:
            return False
    return True


def collect_embeddings(loader, head, args, device, use_amp):
    collectors = {
        "target_raw": [],
        "target_ctx": [],
    }
    counts = {}
    selected_class_ids = select_class_subset(head, args)

    for episode_index, raw_task_dict in enumerate(loader):
        if episode_index >= args.max_episodes:
            break

        task_dict = move_episode_to_device(raw_task_dict, device)
        support_raw, target_raw = extract_raw_features(head, task_dict, use_amp)
        target_ctx = enhance_with_context(head, target_raw, use_amp)

        if args.include_support:
            support_ctx = enhance_with_context(head, support_raw, use_amp)
            support_vectors = reduce_video_features(support_ctx)
            support_labels = task_dict["real_support_labels"].long().cpu().tolist()
            for vec, label in zip(support_vectors.cpu().numpy(), support_labels):
                if selected_class_ids is not None and int(label) not in selected_class_ids:
                    continue
                key = ("support", int(label))
                if counts.get(key, 0) >= args.max_points_per_class:
                    continue
                counts[key] = counts.get(key, 0) + 1
                record = {
                    "embedding": vec,
                    "label_id": int(label),
                    "source": "support",
                    "episode_index": episode_index,
                }
                collectors["target_ctx"].append(record)

        target_raw_vec = reduce_video_features(target_raw)
        target_ctx_vec = reduce_video_features(target_ctx)
        target_labels = task_dict["real_target_labels"].long().cpu().tolist()

        for raw_vec, ctx_vec, label in zip(
            target_raw_vec.cpu().numpy(),
            target_ctx_vec.cpu().numpy(),
            target_labels,
        ):
            if selected_class_ids is not None and int(label) not in selected_class_ids:
                continue
            key = ("target", int(label))
            if counts.get(key, 0) >= args.max_points_per_class:
                continue
            counts[key] = counts.get(key, 0) + 1
            base = {
                "label_id": int(label),
                "source": "target",
                "episode_index": episode_index,
            }
            collectors["target_raw"].append(dict(base, embedding=raw_vec))
            collectors["target_ctx"].append(dict(base, embedding=ctx_vec))

        if is_collection_complete(
            counts,
            selected_class_ids,
            args.include_support,
            args.max_points_per_class,
        ):
            break

    if len(collectors["target_raw"]) == 0:
        raise RuntimeError("No embeddings were collected from the selected split.")

    return collectors, selected_class_ids


def build_mode_records(collectors, head, args):
    records = {
        "target_raw": list(collectors["target_raw"]),
        "target_ctx": list(collectors["target_ctx"]),
    }

    if "target_ctx_text" in args.feature_modes:
        records["target_ctx_text"] = list(collectors["target_ctx"])
        seen_labels = sorted({item["label_id"] for item in collectors["target_ctx"]})
        text_bank, class_names = resolve_text_bank(head, args.split)
        for label_id in seen_labels:
            if label_id >= len(class_names):
                continue
            records["target_ctx_text"].append(
                {
                    "embedding": text_bank[label_id].cpu().numpy(),
                    "label_id": int(label_id),
                    "source": "text",
                    "episode_index": -1,
                }
            )

    return {mode: records[mode] for mode in args.feature_modes}


def resolve_class_name(head, split, label_id):
    names = head.class_real_train if split == "train" else head.class_real_test
    if 0 <= label_id < len(names):
        return names[label_id]
    return str(label_id)


def prepare_arrays(mode_records, head, split):
    embeddings = np.stack([item["embedding"] for item in mode_records], axis=0).astype(np.float32)
    labels = np.array([item["label_id"] for item in mode_records], dtype=np.int64)
    sources = np.array([item["source"] for item in mode_records])
    episodes = np.array([item["episode_index"] for item in mode_records], dtype=np.int64)
    names = np.array([resolve_class_name(head, split, int(label)) for label in labels])
    return embeddings, labels, sources, episodes, names


def run_tsne(embeddings, args):
    if embeddings.shape[0] < 3:
        raise RuntimeError("Need at least 3 points to run t-SNE.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    if embeddings.shape[1] > 2 and embeddings.shape[0] > 3:
        pca_dim = min(args.pca_dim, embeddings.shape[1], embeddings.shape[0] - 1)
        if pca_dim >= 2:
            embeddings = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(embeddings)

    perplexity = min(float(args.perplexity), float(embeddings.shape[0] - 1))
    perplexity = max(1.0, perplexity)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    )
    return tsne.fit_transform(embeddings), perplexity


def save_points_csv(path, points, labels, names, sources, episodes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label_id", "label_name", "source", "episode_index"])
        for idx in range(points.shape[0]):
            writer.writerow(
                [points[idx, 0], points[idx, 1], int(labels[idx]), names[idx], sources[idx], int(episodes[idx])]
            )


def plot_scatter(path, points, labels, sources, args):
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("gist_ncar", max(len(unique_labels), 1))
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(11, 9))
    for source, marker, size, alpha in [
        ("target", "o", args.point_size, args.alpha),
        ("support", "^", args.point_size * 1.1, min(args.alpha + 0.08, 0.9)),
        ("text", "*", args.point_size * 5.0, 1.0),
    ]:
        mask = sources == source
        if not np.any(mask):
            continue
        edgecolors = "black" if source == "text" else "none"
        linewidths = 0.7 if source == "text" else 0.0
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            c=[color_map[int(label)] for label in labels[mask]],
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )

    plt.title("Few-shot t-SNE")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_mode_outputs(mode, mode_records, head, args, save_root):
    mode_dir = os.path.join(save_root, mode)
    os.makedirs(mode_dir, exist_ok=True)

    embeddings, labels, sources, episodes, names = prepare_arrays(mode_records, head, args.split)
    points, perplexity = run_tsne(embeddings, args)

    np.savez_compressed(
        os.path.join(mode_dir, "embeddings.npz"),
        embeddings=embeddings,
        points_2d=points,
        labels=labels,
        label_names=names,
        sources=sources,
        episodes=episodes,
    )
    save_points_csv(os.path.join(mode_dir, "tsne_points.csv"), points, labels, names, sources, episodes)
    plot_scatter(os.path.join(mode_dir, "scatter.png"), points, labels, sources, args)

    label_map = {
        str(int(label)): resolve_class_name(head, args.split, int(label))
        for label in sorted(set(labels.tolist()))
    }
    with open(os.path.join(mode_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    metadata = {
        "config": args.cfg,
        "checkpoint": args.checkpoint if args.checkpoint else getattr(head.args.TEST, "CHECKPOINT_FILE_PATH", ""),
        "split": args.split,
        "mode": mode,
        "feature_dim": int(embeddings.shape[1]),
        "num_points": int(embeddings.shape[0]),
        "num_classes": int(len(set(labels.tolist()))),
        "perplexity": float(perplexity),
        "include_support": bool(args.include_support),
        "max_episodes": int(args.max_episodes),
        "max_points_per_class": int(args.max_points_per_class),
        "requested_num_classes": int(args.num_classes),
        "class_seed": int(args.class_seed if args.class_seed is not None else args.seed),
    }
    with open(os.path.join(mode_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args, cfg_opts = parse_args()
    cfg = load_cfg(args, cfg_opts)
    prepare_cfg(cfg, args)
    set_seed(args.seed)
    ensure_checkpoint_available(cfg)

    device = torch.device("cuda")
    use_amp = bool(getattr(cfg.TRAIN, "USE_AMP", False))

    model, model_ema = build_model(cfg)
    cu.load_test_checkpoint(cfg, model, model_ema)
    model.eval()

    head = get_few_shot_head(model)
    loader = build_loader(cfg, args.split)
    collectors, selected_class_ids = collect_embeddings(loader, head, args, device, use_amp)
    mode_records = build_mode_records(collectors, head, args)

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else "auto_checkpoint"
    save_root = os.path.join(
        args.save_dir,
        os.path.splitext(os.path.basename(args.cfg))[0],
        args.split,
        ckpt_name,
    )
    os.makedirs(save_root, exist_ok=True)

    for mode, records in mode_records.items():
        save_mode_outputs(mode, records, head, args, save_root)

    summary = {mode: len(records) for mode, records in mode_records.items()}
    if selected_class_ids is not None:
        summary["selected_class_ids"] = sorted(int(x) for x in selected_class_ids)
        summary["selected_class_names"] = [
            resolve_class_name(head, args.split, int(x)) for x in sorted(selected_class_ids)
        ]
    with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved t-SNE outputs to: {save_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
