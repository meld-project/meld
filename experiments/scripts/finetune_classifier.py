#!/usr/bin/env python3
"""
微调分类器基线 — FT-BERT 和 FT-Qwen3

全参数微调预训练模型 + 线性分类头，验证集早停。
在 Time-OOD 和 Family-OOD 上评估。
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
LOGGER = logging.getLogger("finetune")


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FinetuneClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, trust_remote_code: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        hidden = outputs.last_hidden_state
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
    model_name: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    max_length: int,
    epochs: int,
    lr: float,
    batch_size: int,
    gradient_accumulation: int,
    device: str,
    trust_remote_code: bool = False,
    seed: int = 42,
    save_encoder_dir: Optional[str] = None,
) -> Dict:
    # 设置随机种子
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

    model = FinetuneClassifier(model_name, trust_remote_code=trust_remote_code).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
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

    # 加载最优模型
    model.load_state_dict(best_state)
    model.to(device)

    # 保存 encoder checkpoint（如果指定）
    if save_encoder_dir:
        save_path = Path(save_encoder_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        # 同时保存完整模型权重
        torch.save(best_state, save_path / "full_model_state.pt")
        LOGGER.info(f"Encoder checkpoint 已保存至 {save_path}")

    # 测试集评估
    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics["best_val_f1"] = best_val_f1
    test_metrics["stopped_epoch"] = epoch + 1 - no_improve
    test_metrics["seed"] = seed

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="微调分类器基线")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--mal_md_dir", required=True)
    parser.add_argument("--ben_md_dir", required=True)
    parser.add_argument("--time_ood_split", required=True)
    parser.add_argument("--family_ood_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_encoder", default=None, help="保存微调后 encoder 的目录")
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

    # === Time-OOD ===
    LOGGER.info("=== Time-OOD 微调 ===")
    with open(args.time_ood_split) as f:
        split = json.load(f)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    test_idx = split["test_indices"]

    time_result = train_and_evaluate(
        model_name=args.model_name,
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
        save_encoder_dir=args.save_encoder,
    )
    LOGGER.info(f"Time-OOD: F1={time_result['macro_f1']:.4f}, TPR@0.1%={time_result['tpr_at_fpr_0.001']:.4f}")

    with open(out / "time_ood_results.json", "w") as f:
        json.dump(time_result, f, indent=2)

    # === Family-OOD ===
    LOGGER.info("=== Family-OOD 微调 ===")
    with open(args.family_ood_split) as f:
        family_spec = json.load(f)

    raw_folds = family_spec["folds"]
    if isinstance(raw_folds, list):
        families = [f["held_out_family"] for f in raw_folds]
        folds_map = {}
        for f in raw_folds:
            folds_map[f["held_out_family"]] = {
                "train_indices": f.get("train_ids", f.get("train_indices", [])),
                "val_indices": f.get("val_ids", f.get("val_indices", [])),
                "test_indices": f.get("test_ids", f.get("test_indices", [])),
            }
    else:
        families = family_spec.get("families", list(raw_folds.keys()))
        folds_map = raw_folds

    family_results = {}
    for family in families:
        LOGGER.info(f"--- 家族: {family} ---")
        fold = folds_map[family]

        result = train_and_evaluate(
            model_name=args.model_name,
            train_texts=[all_texts[i] for i in fold["train_indices"]],
            train_labels=labels[np.array(fold["train_indices"])],
            val_texts=[all_texts[i] for i in fold["val_indices"]],
            val_labels=labels[np.array(fold["val_indices"])],
            test_texts=[all_texts[i] for i in fold["test_indices"]],
            test_labels=labels[np.array(fold["test_indices"])],
            max_length=args.max_length,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            device=device,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
        )
        family_results[family] = result
        LOGGER.info(f"  {family}: F1={result['macro_f1']:.4f}")

    with open(out / "family_ood_results.json", "w") as f:
        json.dump(family_results, f, indent=2)

    LOGGER.info("微调基线全部完成！")


if __name__ == "__main__":
    main()
