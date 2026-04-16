#!/usr/bin/env python3

"""Few-shot Grad-CAM visualization script for CLIP-FSAR.

This script adds a standalone visualization entry under ``runs/`` and does not
modify the training or testing pipeline. It supports the two CLIP visual
backbones used by this project:

- RN50: Grad-CAM on ``head.backbone.layer4[-1]``
- ViT-B/16: spatial Grad-CAM on ``head.backbone.conv1``

The script reuses the project's config loading, model building, checkpoint
loading and few-shot episode loader, then visualizes one query sample from one
episode.
"""

import argparse
import json
import os
import random
import sys
from collections import deque
from types import MethodType

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.curdir))

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

import utils.checkpoint as cu  # noqa: E402
from datasets.base.builder import build_loader  # noqa: E402
from models.base.builder import build_model  # noqa: E402
from utils.config import Config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for CLIP-FSAR few-shot episodes."
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Path to the config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. If omitted, follows the project's normal test loading logic.",
    )
    parser.add_argument(
        "--backbone",
        default="auto",
        choices=["auto", "RN50", "ViT-B/16"],
        help="Backbone override. 'auto' uses the value from the config.",
    )
    parser.add_argument(
        "--save_dir",
        default="output/gradcam",
        help="Directory used to save heatmaps.",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Which few-shot episode in the test loader to visualize.",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=0,
        help="Which query sample inside the selected episode to visualize.",
    )
    parser.add_argument(
        "--score_mode",
        default="pred",
        choices=["pred", "gt", "index"],
        help="Target score for backprop: predicted class, ground-truth class, or a fixed class index.",
    )
    parser.add_argument(
        "--class_index",
        type=int,
        default=None,
        help="Episode-local class index used when score_mode=index.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=-1,
        help="If >=0, save only one frame. If -1, save all frames.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for this visualization script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
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
        raise RuntimeError(
            "当前项目的 CLIP few-shot 头在初始化时直接走 CUDA，Grad-CAM 脚本需要可用 GPU。"
        )

    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.SHARD_ID = 0
    cfg.DATA_LOADER.NUM_WORKERS = args.num_workers
    cfg.TEST.BATCH_SIZE = 1

    if args.backbone != "auto":
        cfg.VIDEO.HEAD.BACKBONE_NAME = args.backbone
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
        raise RuntimeError("顶层 model 未找到 head，无法定位 few-shot 视觉骨干。")
    head = net.head
    if not hasattr(head, "backbone") or not hasattr(head, "get_feats"):
        raise RuntimeError("当前 head 不具备 few-shot CLIP backbone/get_feats 接口。")
    return head


def resolve_target_layer(head):
    backbone_name = head.args.VIDEO.HEAD.BACKBONE_NAME
    if backbone_name == "RN50":
        layer = head.backbone.layer4[-1]
        layer_name = "head.backbone.layer4[-1]"
    elif backbone_name == "ViT-B/16":
        layer = head.backbone.conv1
        layer_name = "head.backbone.conv1"
    else:
        raise ValueError(
            f"暂不支持的 backbone: {backbone_name}。当前脚本仅支持 RN50 和 ViT-B/16。"
        )
    return layer, layer_name


class GradCAMRecorder:
    def __init__(self, owner):
        self.owner = owner
        self.activations = {"support": [], "target": [], "unknown": []}
        self.gradients = {"support": [], "target": [], "unknown": []}
        self._stream_stack = deque()

    def reset(self):
        for key in self.activations:
            self.activations[key].clear()
            self.gradients[key].clear()
        self._stream_stack.clear()

    def forward_hook(self, module, inputs, output):
        stream = getattr(self.owner, "_gradcam_stream", "unknown")
        if stream not in self.activations:
            stream = "unknown"
        self.activations[stream].append(output)
        self._stream_stack.append(stream)

    def backward_hook(self, module, grad_input, grad_output):
        stream = self._stream_stack.pop() if self._stream_stack else "unknown"
        if stream not in self.gradients:
            stream = "unknown"
        self.gradients[stream].append(grad_output[0])


def patch_get_feats_for_stream_tracking(head):
    if getattr(head, "_gradcam_get_feats_patched", False):
        return

    def patched_get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        if self.training:
            self._gradcam_stream = "support"
            support_features = self.backbone(support_images).squeeze()
            self._gradcam_stream = "target"
            target_features = self.backbone(target_images).squeeze()
            self._gradcam_stream = None

            dim = int(support_features.shape[1])
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_text = None
        else:
            self._gradcam_stream = "support"
            support_features = self.backbone(support_images).squeeze()
            self._gradcam_stream = "target"
            target_features = self.backbone(target_images).squeeze()
            self._gradcam_stream = None

            dim = int(target_features.shape[1])
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_text = self.text_features_test[support_real_class.long()]

        return support_features, target_features, support_features_text

    head.get_feats = MethodType(patched_get_feats, head)
    head._gradcam_get_feats_patched = True


def fetch_episode(loader, episode_index, device):
    for idx, task_dict in enumerate(loader):
        if idx != episode_index:
            continue

        episode = {}
        for key, value in task_dict.items():
            if torch.is_tensor(value):
                episode[key] = value[0].to(device, non_blocking=True)
            else:
                episode[key] = value
        return episode

    raise IndexError(f"episode_index={episode_index} 超出 train loader 范围。")


def select_score(logits, task_dict, query_index, score_mode, class_index):
    pred_class = int(logits[query_index].argmax(dim=0).item())
    gt_class = int(task_dict["target_labels"][query_index].item())

    if score_mode == "pred":
        target_class = pred_class
    elif score_mode == "gt":
        target_class = gt_class
    else:
        if class_index is None:
            raise ValueError("score_mode=index 时必须提供 --class_index。")
        target_class = int(class_index)

    if target_class < 0 or target_class >= logits.shape[1]:
        raise IndexError(
            f"目标类别索引越界: {target_class}，当前 episode 类别数为 {logits.shape[1]}。"
        )

    score = logits[query_index, target_class]
    return score, pred_class, gt_class, target_class


def compute_cam_from_records(recorder, query_count, frame_count, output_size):
    if not recorder.activations["target"]:
        raise RuntimeError("未捕获到 target 分支的前向激活。")
    if not recorder.gradients["target"]:
        raise RuntimeError("未捕获到 target 分支的反向梯度。")

    activation = recorder.activations["target"][-1].float()
    gradient = recorder.gradients["target"][0].float()

    if activation.dim() != 4 or gradient.dim() != 4:
        raise RuntimeError(
            f"当前 hook 张量维度不是 4D，activation={tuple(activation.shape)}, gradient={tuple(gradient.shape)}。"
        )

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1))
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    if cam.shape[0] != query_count * frame_count:
        raise RuntimeError(
            "CAM 帧数与 target_set 不匹配: "
            f"cam_batch={cam.shape[0]}, query_count={query_count}, frame_count={frame_count}"
        )

    cam = cam.reshape(query_count, frame_count, output_size, output_size)
    cam_min = cam.amin(dim=(-2, -1), keepdim=True)
    cam_max = cam.amax(dim=(-2, -1), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return cam.detach().cpu()


def denormalize_frames(task_dict, cfg):
    frame_count = int(cfg.DATA.NUM_INPUT_FRAMES)
    crop_size = int(cfg.DATA.TEST_CROP_SIZE)
    query_count = int(task_dict["target_labels"].shape[0])

    frames = task_dict["target_set"].detach().cpu()
    frames = frames.reshape(query_count, frame_count, 3, crop_size, crop_size)

    mean = torch.tensor(cfg.DATA.MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg.DATA.STD).view(1, 1, 3, 1, 1)
    frames = (frames * std + mean).clamp(0.0, 1.0)
    return frames


def resolve_class_name(head, real_class_id):
    if hasattr(head, "class_real_test") and 0 <= real_class_id < len(head.class_real_test):
        return head.class_real_test[real_class_id]
    if hasattr(head, "class_real_train") and 0 <= real_class_id < len(head.class_real_train):
        return head.class_real_train[real_class_id]
    return str(real_class_id)


def make_panel(rgb_image, cam_map):
    cmap = cm.get_cmap("jet")

    img_uint8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    heat_rgb = cmap(cam_map)[..., :3]
    heat_uint8 = np.clip(heat_rgb * 255.0, 0, 255).astype(np.uint8)
    overlay = np.clip(
        0.5 * img_uint8.astype(np.float32) + 0.5 * heat_uint8.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return np.concatenate([img_uint8, heat_uint8, overlay], axis=1)


def save_outputs(save_root, frames, cams, frame_index):
    os.makedirs(save_root, exist_ok=True)

    selected_indices = [frame_index] if frame_index >= 0 else list(range(frames.shape[0]))

    strips = []
    for idx in selected_indices:
        rgb = frames[idx].permute(1, 2, 0).numpy()
        cam = cams[idx].numpy()
        panel = make_panel(rgb, cam)
        strips.append(panel)
        Image.fromarray(panel).save(os.path.join(save_root, f"frame_{idx:02d}.png"))

    montage = np.concatenate(strips, axis=0)
    Image.fromarray(montage).save(os.path.join(save_root, "montage.png"))
    np.save(os.path.join(save_root, "cam.npy"), cams.numpy())


def main():
    args, cfg_opts = parse_args()
    cfg = load_cfg(args, cfg_opts)
    prepare_cfg(cfg, args)
    set_seed(args.seed)
    use_amp = bool(getattr(cfg.TRAIN, "USE_AMP", False))

    device = torch.device("cuda")

    model, model_ema = build_model(cfg)
    cu.load_test_checkpoint(cfg, model, model_ema)
    model.eval()

    head = get_few_shot_head(model)
    # The script is hardcoded to visualize train episodes. Keep eval mode for
    # stable Grad-CAM, but redirect eval-time class/text banks to the train split.
    if hasattr(head, "text_features_train") and hasattr(head, "text_features_test"):
        head.text_features_test = head.text_features_train
    if hasattr(head, "class_real_train") and hasattr(head, "class_real_test"):
        head.class_real_test = head.class_real_train
    patch_get_feats_for_stream_tracking(head)
    target_layer, layer_name = resolve_target_layer(head)

    recorder = GradCAMRecorder(head)
    forward_handle = target_layer.register_forward_hook(recorder.forward_hook)
    backward_handle = target_layer.register_full_backward_hook(recorder.backward_hook)

    try:
        loader = build_loader(cfg, "train")
        task_dict = fetch_episode(loader, args.episode_index, device)

        query_count = int(task_dict["target_labels"].shape[0])
        frame_count = int(cfg.DATA.NUM_INPUT_FRAMES)
        crop_size = int(cfg.DATA.TEST_CROP_SIZE)

        if args.query_index < 0 or args.query_index >= query_count:
            raise IndexError(
                f"query_index={args.query_index} 越界，当前 episode query 数为 {query_count}。"
            )

        recorder.reset()
        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            model_dict = model(task_dict)
            logits = model_dict["logits"]

        score, pred_class, gt_class, target_class = select_score(
            logits,
            task_dict,
            args.query_index,
            args.score_mode,
            args.class_index,
        )
        score.float().backward()

        cams = compute_cam_from_records(
            recorder,
            query_count=query_count,
            frame_count=frame_count,
            output_size=crop_size,
        )
        frames = denormalize_frames(task_dict, cfg)

        episode_class_map = task_dict["batch_class_list"].long().detach().cpu()
        pred_real_class = int(episode_class_map[pred_class].item())
        gt_real_class = int(task_dict["real_target_labels"][args.query_index].item())
        target_real_class = int(episode_class_map[target_class].item())

        pred_name = resolve_class_name(head, pred_real_class)
        gt_name = resolve_class_name(head, gt_real_class)
        target_name = resolve_class_name(head, target_real_class)

        save_root = os.path.join(
            args.save_dir,
            os.path.splitext(os.path.basename(args.cfg))[0],
            f"episode_{args.episode_index:04d}",
            f"query_{args.query_index:03d}",
        )

        save_outputs(
            save_root=save_root,
            frames=frames[args.query_index],
            cams=cams[args.query_index],
            frame_index=args.frame_index,
        )

        metadata = {
            "config": args.cfg,
            "checkpoint": cfg.TEST.CHECKPOINT_FILE_PATH,
            "backbone_name": head.args.VIDEO.HEAD.BACKBONE_NAME,
            "use_amp": use_amp,
            "target_layer": layer_name,
            "episode_index": int(args.episode_index),
            "query_index": int(args.query_index),
            "frame_index": int(args.frame_index),
            "score_mode": args.score_mode,
            "pred_episode_class": pred_class,
            "gt_episode_class": gt_class,
            "target_episode_class": target_class,
            "pred_real_class": pred_real_class,
            "gt_real_class": gt_real_class,
            "target_real_class": target_real_class,
            "pred_name": pred_name,
            "gt_name": gt_name,
            "target_name": target_name,
            "save_root": save_root,
        }

        with open(os.path.join(save_root, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Grad-CAM saved to: {save_root}")
        print(f"Backbone: {metadata['backbone_name']}")
        print(f"Target layer: {metadata['target_layer']}")
        print(f"Pred: [{pred_real_class}] {pred_name}")
        print(f"GT:   [{gt_real_class}] {gt_name}")
        print(f"CAM target: [{target_real_class}] {target_name}")
    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":
    main()



