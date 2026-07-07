#!/usr/bin/env python3
"""
方案B：在指定中间层加分类头微调

标准微调永远在最终层加分类头。本实验验证：如果把分类头放在中间层（如 L21），
微调性能是否会进一步提升？

用法:
    # 在 L21 加分类头微调
    python finetune_at_layer.py \
        --model_name models/Qwen3-0.6B \
        --target_layer 21 \
        --mal_md_dir /data/meld-data/cape_reports_malicious_md \
        --ben_md_dir /data/meld-data/cape_reports_benign_md \
        --time_ood_split experiments/splits/time_ood_split.json \
        --output_dir experiments/results_v2/supplementary/plan_b/ft_at_L21 \
        --gpu 0

    # 对照：标准最终层微调 (L28)
    python finetune_at_layer.py \
        --model_name models/Qwen3-0.6B \
        --target_layer 28 \
        ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("finetune_at_layer")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FinetuneAtLayer(nn.Module):
    """在指定层加分类头的微调模型。

    支持两种模式：
    - freeze_after=True (设计A)：冻结 target_layer 之后的层，只更新 L1-target_layer + 分类头
    - freeze_after=False (设计B)：全部层参与梯度更新，但分类头接在 target_layer
    """

    def __init__(self, model_name: str, target_layer: int, num_classes: int = 2,
                 trust_remote_code: bool = False, freeze_after: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            output_hidden_states=True,
        )
        self.target_layer = target_layer
        self.freeze_after = freeze_after
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

        total_layers = self.encoder.config.num_hidden_layers
        LOGGER.info(f"模型总层数: {total_layers}, 分类头在: L{target_layer}, 冻结后层: {freeze_after}")

        # 设计A：冻结 target_layer 之后的所有层
        if freeze_after and target_layer < total_layers:
            frozen_count = 0
            # 获取 transformer 层列表（兼容不同模型架构）
            if hasattr(self.encoder, 'layers'):
                layer_modules = self.encoder.layers
            elif hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'layers'):
                layer_modules = self.encoder.model.layers
            elif hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                layer_modules = self.encoder.encoder.layer
            else:
                layer_modules = None
                LOGGER.warning("无法识别层结构，改为按参数名冻结")

            if layer_modules is not None:
                for i, layer in enumerate(layer_modules):
                    if i >= target_layer:  # 层索引从0开始，target_layer从1开始对应索引target_layer
                        for param in layer.parameters():
                            param.requires_grad = False
                            frozen_count += 1
            else:
                # 回退方案：按参数名匹配冻结
                for name, param in self.encoder.named_parameters():
                    # 匹配类似 layers.21. / layer.21. 的参数名
                    import re
                    match = re.search(r'layers?\.(\d+)\.', name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx >= target_layer:
                            param.requires_grad = False
                            frozen_count += 1

            # 同时冻结 final layernorm（如果有）
            for name, param in self.encoder.named_parameters():
                if 'final_layernorm' in name or 'norm' == name.split('.')[-1]:
                    # 只冻结最外层的 norm，不冻结层内的 norm
                    if name.count('.') <= 2:
                        param.requires_grad = False
                        frozen_count += 1

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            LOGGER.info(f"冻结了 {frozen_count} 个参数张量，可训练参数: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_states[0] = embedding层输出, hidden_states[i] = 第i层输出
        hidden = outputs.hidden_states[self.target_layer]
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def tpr_at_fpr(y_true, y_prob, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = fpr <= target_fpr
    return float(tpr[valid][-1]) if valid.any() else 0.0


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tpr_at_fpr_0.001": tpr_at_fpr(y_true, y_prob, 0.001),
        "tpr_at_fpr_0.005": tpr_at_fpr(y_true, y_prob, 0.005),
        "tpr_at_fpr_0.01": tpr_at_fpr(y_true, y_prob, 0.01),
    }


def train_and_evaluate(
    model_name, target_layer, train_texts, train_labels,
    val_texts, val_labels, test_texts, test_labels,
    max_length, epochs, lr, batch_size, gradient_accumulation,
    device, trust_remote_code=False, seed=42, freeze_after=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=2)

    model = FinetuneAtLayer(model_name, target_layer, trust_remote_code=trust_remote_code,
                            freeze_after=freeze_after).to(device)
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_state = None
    patience = 2
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (L{target_layer})")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / gradient_accumulation
            loss.backward()
            total_loss += loss.item() * gradient_accumulation

            if (step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)
        LOGGER.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_F1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                LOGGER.info(f"早停于 Epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics["best_val_f1"] = best_val_f1
    test_metrics["stopped_epoch"] = epoch + 1 - no_improve
    test_metrics["target_layer"] = target_layer
    test_metrics["seed"] = seed
    test_metrics["freeze_after"] = freeze_after

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="方案B：中间层分类头微调")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--target_layer", type=int, required=True, help="分类头所在层号 (1-28)")
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--time_ood_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--freeze_after", action="store_true",
                        help="设计A：冻结 target_layer 之后的层（默认为设计B：全部层参与更新）")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # 加载文本
    mal_paths = sorted(Path(args.mal_md_dir).glob("*.md"))
    ben_paths = sorted(Path(args.ben_md_dir).glob("*.md"))
    all_texts = [p.read_text(encoding="utf-8", errors="replace") for p in mal_paths] + \
                [p.read_text(encoding="utf-8", errors="replace") for p in ben_paths]
    labels = np.array([1] * len(mal_paths) + [0] * len(ben_paths))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Time-OOD
    with open(args.time_ood_split) as f:
        split = json.load(f)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    test_idx = split["test_indices"]

    mode = "设计A(截断)" if args.freeze_after else "设计B(全更新)"
    LOGGER.info(f"=== Time-OOD 微调 @ L{args.target_layer} [{mode}] ===")
    result = train_and_evaluate(
        model_name=args.model_name,
        target_layer=args.target_layer,
        train_texts=[all_texts[i] for i in train_idx],
        train_labels=labels[train_idx],
        val_texts=[all_texts[i] for i in val_idx],
        val_labels=labels[val_idx],
        test_texts=[all_texts[i] for i in test_idx],
        test_labels=labels[test_idx],
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        device=device,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        freeze_after=args.freeze_after,
    )

    LOGGER.info(f"FT@L{args.target_layer}: F1={result['macro_f1']:.4f}, TPR@0.1%={result['tpr_at_fpr_0.001']:.4f}")

    with open(out / "time_ood_results.json", "w") as f:
        json.dump(result, f, indent=2)

    LOGGER.info(f"结果已保存至 {out}")


if __name__ == "__main__":
    main()
