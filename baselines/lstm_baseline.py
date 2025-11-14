"""LSTM baseline model for malware detection.

LSTM processes text sequences (from Markdown reports) with bidirectional LSTM layers.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# W&B integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

LOGGER = logging.getLogger("lstm_baseline")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    """Configuration for LSTM baseline training."""
    mal_dir: Optional[str] = None  # 用于 holdout/cv/time_ood 模式
    benign_dir: Optional[str] = None  # 用于 holdout/cv/time_ood 模式
    train_dir: Optional[str] = None  # 用于 dir 模式
    val_dir: Optional[str] = None  # 用于 dir 模式
    test_dir: Optional[str] = None  # 用于 dir 模式
    split_mode: str = "holdout"  # holdout / cv / dir / time_ood
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    seeds: Optional[List[int]] = None
    limit: Optional[int] = None
    n_splits: int = 5
    target_fpr: float = 0.01
    
    # Time-OOD specific
    malicious_manifest_path: Optional[str] = None  # 用于 time_ood 模式
    benign_manifest_path: Optional[str] = None  # 用于 time_ood 模式
    train_end_date: Optional[str] = None  # 用于 time_ood 模式
    val_end_date: Optional[str] = None  # 用于 time_ood 模式
    
    # LSTM specific
    vocab_size: int = 10000
    max_length: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    embedding_dim: int = 128
    
    # Training
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    out: Optional[str] = None
    save_raw: bool = False
    progress: bool = False
    
    # W&B 配置
    use_wandb: bool = False
    wandb_project: str = "lec-experiments"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    run_tag: str = ""
    
    config_path: Optional[str] = field(default=None, repr=False, compare=False)
    
    def validate(self) -> None:
        if self.split_mode not in {"holdout", "cv", "dir", "time_ood"}:
            raise ValueError("split_mode 必须是 holdout/cv/dir/time_ood")
        
        if self.split_mode == "dir":
            # dir 模式需要 train_dir, val_dir, test_dir
            for key, name in [
                (self.train_dir, "train_dir"),
                (self.val_dir, "val_dir"),
                (self.test_dir, "test_dir"),
            ]:
                if not key:
                    raise ValueError(f"dir 模式需要提供 {name}")
                path = Path(key)
                if not path.exists() or not path.is_dir():
                    raise FileNotFoundError(f"{name} 目录不存在：{path}")
        elif self.split_mode == "time_ood":
            # time_ood 模式需要 manifest 路径和目录
            required_fields = [
                ("malicious_manifest_path", "malicious_manifest_path"),
                ("benign_manifest_path", "benign_manifest_path"),
                ("mal_dir", "mal_dir"),
                ("benign_dir", "benign_dir"),
            ]
            for attr_name, field_name in required_fields:
                value = getattr(self, attr_name, None)
                if not value:
                    raise ValueError(f"time_ood 模式需要提供 {field_name}")
                if attr_name.endswith("_path"):
                    path = Path(value)
                    if not path.exists() or not path.is_file():
                        raise FileNotFoundError(f"{field_name} 文件不存在：{path}")
                elif attr_name.endswith("_dir"):
                    path = Path(value)
                    if not path.exists() or not path.is_dir():
                        raise FileNotFoundError(f"{field_name} 目录不存在：{path}")
        else:
            # holdout/cv 模式：必须使用 mal_dir + benign_dir
            if not self.mal_dir or not self.benign_dir:
                raise ValueError("holdout/cv 模式需要提供 mal_dir 和 benign_dir")
            
            # 验证 mal_dir 和 benign_dir
            mal_path = Path(self.mal_dir)
            ben_path = Path(self.benign_dir)
            if not mal_path.exists() or not mal_path.is_dir():
                raise FileNotFoundError(f"mal_dir 目录不存在：{mal_path}")
            if not ben_path.exists() or not ben_path.is_dir():
                raise FileNotFoundError(f"benign_dir 目录不存在：{ben_path}")
            
            if self.split_mode == "holdout":
                if not (0 < self.val_ratio < 1) or not (0 < self.test_ratio < 1):
                    raise ValueError("val_ratio/test_ratio 必须在 (0,1) 内")
                if self.val_ratio + self.test_ratio >= 0.95:
                    raise ValueError("val_ratio + test_ratio 过大")
        
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit 至少为 2")
        if not (0 < self.target_fpr < 1):
            raise ValueError("target_fpr 必须在 (0,1) 内")
        if self.seeds:
            cleaned = sorted(dict.fromkeys(int(s) for s in self.seeds))
            if not cleaned:
                raise ValueError("seeds 不能为空")
            self.seeds = cleaned


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #


def load_md_reports(md_dir: str, limit: Optional[int] = None) -> Tuple[List[str], List[int], List[str]]:
    """Load Markdown reports with labels."""
    md_path = Path(md_dir)
    if not md_path.exists():
        raise FileNotFoundError(f"目录不存在: {md_dir}")
    
    texts = []
    labels = []
    ids = []
    
    # 尝试多种目录结构
    # 1. black/white 子目录
    mal_dir = md_path / "black"
    ben_dir = md_path / "white"
    
    # 2. cape_reports_malicious_md / cape_reports_benign_md
    if not mal_dir.exists():
        mal_dir = md_path / "cape_reports_malicious_md"
    if not ben_dir.exists():
        ben_dir = md_path / "cape_reports_benign_md"
    
    # Load malicious samples
    if mal_dir.exists():
        mal_files = sorted(mal_dir.glob("*.md"))
        if limit:
            # 分层抽样：确保恶意和良性样本数量平衡
            half_limit = limit // 2
            mal_files = mal_files[:half_limit]
        for f in mal_files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                texts.append(text)
                labels.append(1)
                ids.append(f.stem)
            except Exception as e:
                LOGGER.warning(f"Failed to load {f}: {e}")
    else:
        LOGGER.warning(f"未找到恶意样本目录: {mal_dir}")
    
    # Load benign samples
    if ben_dir.exists():
        ben_files = sorted(ben_dir.glob("*.md"))
        if limit:
            # 分层抽样：确保恶意和良性样本数量平衡
            half_limit = limit // 2
            ben_files = ben_files[:half_limit]
        for f in ben_files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                texts.append(text)
                labels.append(0)
                ids.append(f.stem)
            except Exception as e:
                LOGGER.warning(f"Failed to load {f}: {e}")
    else:
        LOGGER.warning(f"未找到良性样本目录: {ben_dir}")
    
    if not texts:
        raise RuntimeError(f"未读取到任何样本，请检查 Markdown 目录: {md_dir}")
    
    return texts, labels, ids


def load_md_reports_from_dir(data_dir: str, progress: bool = False) -> Tuple[List[str], List[int], List[str]]:
    """Load Markdown reports from a directory with black/white subdirectories.
    
    Args:
        data_dir: Directory containing black/ and white/ subdirectories
        progress: Show progress bar
    
    Returns:
        Tuple of (texts, labels, ids)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    
    # Try black/white structure
    mal_dir = data_path / "black"
    ben_dir = data_path / "white"
    
    # Try alternative names
    if not mal_dir.exists():
        mal_dir = data_path / "cape_reports_malicious_md"
    if not ben_dir.exists():
        ben_dir = data_path / "cape_reports_benign_md"
    
    mal_files = sorted(mal_dir.glob("*.md")) if mal_dir.exists() else []
    ben_files = sorted(ben_dir.glob("*.md")) if ben_dir.exists() else []
    
    texts = []
    labels = []
    ids = []
    
    all_files = [(f, 1) for f in mal_files] + [(f, 0) for f in ben_files]
    
    iterator = all_files
    if progress:
        iterator = tqdm(iterator, desc=f"Loading from {data_dir}", total=len(all_files))
    
    for md_file, label in iterator:
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            texts.append(text)
            labels.append(label)
            ids.append(md_file.stem)
        except Exception as e:
            LOGGER.warning(f"Failed to load {md_file}: {e}")
    
    LOGGER.info(f"从 {data_dir} 加载了 {len(texts)} 个报告 (恶意: {len(mal_files)}, 良性: {len(ben_files)})")
    
    return texts, labels, ids


# --------------------------------------------------------------------------- #
# LSTM 模型
# --------------------------------------------------------------------------- #


class LSTMModel(nn.Module):
    """Bidirectional LSTM model for sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x, lengths=None):
        # x: [B, L]
        embedded = self.embedding(x)  # [B, L, E]
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # [B, L, H*2]
        
        # Use last hidden state
        # h_n: [num_layers*2, B, H] (bidirectional)
        last_hidden = h_n[-1]  # [B, H] (forward)
        last_hidden_backward = h_n[-2] if self.lstm.bidirectional else None  # [B, H] (backward)
        if last_hidden_backward is not None:
            # Concatenate forward and backward
            combined = torch.cat([last_hidden, last_hidden_backward], dim=1)  # [B, H*2]
        else:
            combined = last_hidden
        
        output = self.dropout(combined)
        output = self.fc(output)  # [B, num_classes]
        return output


# --------------------------------------------------------------------------- #
# 文本预处理
# --------------------------------------------------------------------------- #


def build_vocab(texts: Sequence[str], vocab_size: int) -> Dict[str, int]:
    """Build vocabulary from texts."""
    word_counter = Counter()
    for text in texts:
        words = text.lower().split()
        word_counter.update(words)
    
    # Most common words + special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in word_counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab


def text_to_sequence(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    """Convert text to sequence of token IDs."""
    words = text.lower().split()
    seq = [vocab.get(word, vocab["<unk>"]) for word in words]
    
    # Pad or truncate
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq = seq + [vocab["<pad>"]] * (max_length - len(seq))
    
    return seq


# --------------------------------------------------------------------------- #
# 评估函数
# --------------------------------------------------------------------------- #


def quantile_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Dict[str, float]:
    """Calculate threshold at target FPR."""
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    if mask_neg.sum() == 0 or mask_pos.sum() == 0:
        raise ValueError("正负样本数量不足，无法计算阈值。")
    neg_scores = y_score[mask_neg]
    tau = float(np.quantile(neg_scores, 1 - target_fpr))
    preds = (y_score >= tau).astype(int)
    fpr = float(((preds == 1) & mask_neg).sum() / mask_neg.sum())
    tpr = float(((preds == 1) & mask_pos).sum() / mask_pos.sum())
    mu_neg, std_neg = float(neg_scores.mean()), float(neg_scores.std(ddof=1) + 1e-8)
    mu_all, std_all = float(y_score.mean()), float(y_score.std(ddof=1) + 1e-8)
    z_neg = (tau - mu_neg) / std_neg
    z_all = (tau - mu_all) / std_all
    return {"tau": tau, "fpr": fpr, "tpr": tpr, "z_neg": float(z_neg), "z_all": float(z_all)}


def stratified_holdout_indices(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test."""
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    val_rel = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


def train_lstm(
    model: LSTMModel,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    progress: bool = False,
) -> List[float]:
    """Train LSTM model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    losses = []
    
    iterator = range(epochs)
    if progress:
        iterator = tqdm(iterator, desc="Training", total=epochs)
    
    for epoch in iterator:
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
    
    return losses


def evaluate_lstm(
    model: LSTMModel,
    data_loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate LSTM model and return predictions and labels."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_probs), np.array(all_labels)


def evaluate_holdout(
    texts: Sequence[str],
    labels: Sequence[int],
    vocab: Dict[str, int],
    cfg: TrainConfig,
) -> Dict:
    """Evaluate with holdout split."""
    labels_arr = np.array(labels, dtype=int)
    train_idx, val_idx, test_idx = stratified_holdout_indices(
        labels_arr, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )
    
    # Convert texts to sequences
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    
    train_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in train_texts]
    val_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in val_texts]
    test_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in test_texts]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_seqs, dtype=torch.long),
        torch.tensor(labels_arr[train_idx], dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_seqs, dtype=torch.long),
        torch.tensor(labels_arr[val_idx], dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_seqs, dtype=torch.long),
        torch.tensor(labels_arr[test_idx], dtype=torch.long),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # Create and train model
    device_obj = torch.device(cfg.device)
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=cfg.embedding_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device_obj)
    
    train_lstm(model, train_loader, cfg.device, cfg.epochs, cfg.learning_rate, cfg.progress)
    
    # Evaluate on validation set
    val_probs, val_labels = evaluate_lstm(model, val_loader, cfg.device)
    id_stats = quantile_threshold(val_labels, val_probs, cfg.target_fpr)
    
    # Evaluate on test set
    test_probs, test_labels = evaluate_lstm(model, test_loader, cfg.device)
    
    # Apply threshold
    preds = (test_probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(test_labels, preds, average="macro"))
    auroc = float(roc_auc_score(test_labels, test_probs))
    aupr = float(average_precision_score(test_labels, test_probs))
    accuracy = float(accuracy_score(test_labels, preds))
    
    return {
        "mode": "holdout",
        "baseline": "lstm",
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "f1_id": float(f1_score(val_labels, (val_probs >= id_stats["tau"]).astype(int), average="macro")),
            "auc_id": float(roc_auc_score(val_labels, val_probs)),
            "tpr_ood": id_stats["tpr"],
            "fpr_ood": id_stats["fpr"],
            "f1_ood": macro_f1,
            "auc_ood": auroc,
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
    }


def evaluate_holdout_with_predefined_indices(
    texts: Sequence[str],
    labels: Sequence[int],
    vocab: Dict[str, int],
    n_train: int,
    n_val: int,
    n_test: int,
    cfg: TrainConfig,
) -> Dict:
    """Evaluate with predefined train/val/test split (dir mode)."""
    labels_arr = np.array(labels, dtype=int)
    
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    
    # Convert texts to sequences
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    
    train_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in train_texts]
    val_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in val_texts]
    test_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in test_texts]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_seqs, dtype=torch.long),
        torch.tensor(labels_arr[train_idx], dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_seqs, dtype=torch.long),
        torch.tensor(labels_arr[val_idx], dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_seqs, dtype=torch.long),
        torch.tensor(labels_arr[test_idx], dtype=torch.long),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # Create and train model
    device_obj = torch.device(cfg.device)
    model = LSTMModel(
        vocab_size=len(vocab),
        embedding_dim=cfg.embedding_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device_obj)
    
    train_lstm(model, train_loader, cfg.device, cfg.epochs, cfg.learning_rate, cfg.progress)
    
    # Evaluate on validation set
    val_probs, val_labels = evaluate_lstm(model, val_loader, cfg.device)
    id_stats = quantile_threshold(val_labels, val_probs, cfg.target_fpr)
    
    # Evaluate on test set
    test_probs, test_labels = evaluate_lstm(model, test_loader, cfg.device)
    
    # Apply threshold
    preds = (test_probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(test_labels, preds, average="macro"))
    auroc = float(roc_auc_score(test_labels, test_probs))
    aupr = float(average_precision_score(test_labels, test_probs))
    accuracy = float(accuracy_score(test_labels, preds))
    
    return {
        "mode": "holdout",
        "baseline": "lstm",
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr_id": id_stats["tpr"],
            "fpr_id": id_stats["fpr"],
            "f1_id": float(f1_score(val_labels, (val_probs >= id_stats["tau"]).astype(int), average="macro")),
            "auc_id": float(roc_auc_score(val_labels, val_probs)),
            "tpr_ood": id_stats["tpr"],
            "fpr_ood": id_stats["fpr"],
            "f1_ood": macro_f1,
            "auc_ood": auroc,
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
        "_test_idx": test_idx,  # Store for visualization
    }


def evaluate_cv(
    texts: Sequence[str],
    labels: Sequence[int],
    vocab: Dict[str, int],
    cfg: TrainConfig,
) -> Dict:
    """Evaluate with cross-validation."""
    labels_arr = np.array(labels, dtype=int)
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    
    all_probs = np.zeros_like(labels_arr, dtype=float)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_arr)):
        # Convert texts to sequences
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        
        train_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in train_texts]
        val_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in val_texts]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_seqs, dtype=torch.long),
            torch.tensor(labels_arr[train_idx], dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_seqs, dtype=torch.long),
            torch.tensor(labels_arr[val_idx], dtype=torch.long),
        )
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        # Create and train model
        device_obj = torch.device(cfg.device)
        model = LSTMModel(
            vocab_size=len(vocab),
            embedding_dim=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device_obj)
        
        train_lstm(model, train_loader, cfg.device, cfg.epochs, cfg.learning_rate, cfg.progress)
        
        # Predict
        val_probs, _ = evaluate_lstm(model, val_loader, cfg.device)
        all_probs[val_idx] = val_probs
    
    # Calculate metrics
    id_stats = quantile_threshold(labels_arr, all_probs, cfg.target_fpr)
    preds = (all_probs >= id_stats["tau"]).astype(int)
    macro_f1 = float(f1_score(labels_arr, preds, average="macro"))
    auroc = float(roc_auc_score(labels_arr, all_probs))
    aupr = float(average_precision_score(labels_arr, all_probs))
    accuracy = float(accuracy_score(labels_arr, preds))
    
    return {
        "mode": "cv",
        "baseline": "lstm",
        "target_fpr": cfg.target_fpr,
        "best": {
            "threshold": id_stats["tau"],
            "tpr": id_stats["tpr"],
            "fpr": id_stats["fpr"],
            "macro_f1": macro_f1,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
        },
    }


# --------------------------------------------------------------------------- #
# W&B 集成
# --------------------------------------------------------------------------- #


def init_wandb(cfg: TrainConfig) -> Optional[str]:
    """Initialize W&B run."""
    if not cfg.use_wandb or not WANDB_AVAILABLE:
        return None
    
    if cfg.wandb_api_key:
        wandb.login(key=cfg.wandb_api_key)
    
    # 结束之前的 run（如果存在），避免并行执行时的 run 名称混乱
    if wandb.run is not None:
        wandb.finish()
    
    run_name = f"lstm_{cfg.split_mode}"
    if cfg.run_tag:
        run_name += f"_{cfg.run_tag}"
    
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config=asdict(cfg),
        reinit=True,  # 允许重新初始化，确保每个基线都有独立的 run
    )
    return wandb.run.id if wandb.run else None


def log_metrics_to_wandb(metrics: Dict, prefix: str = "") -> None:
    """Log metrics to W&B."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    wandb_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            wandb_metrics[f"{prefix}_{key}" if prefix else key] = value
    
    try:
        wandb.log(wandb_metrics)
    except Exception:
        # W&B run 可能已结束，静默忽略
        pass


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def load_config_file(path: Optional[str]) -> Dict:
    """Load configuration from JSON file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象，当前: {type(data)}")
    return data


def aggregate_seed_summaries(cfg: TrainConfig, summaries: List[Dict]) -> Dict:
    """Aggregate results across multiple seeds."""
    seeds = [s["seed"] for s in summaries]
    best_keys = set()
    for summary in summaries:
        best = summary.get("best", {})
        for key, value in best.items():
            if isinstance(value, (int, float)):
                best_keys.add(key)
    
    best_mean: Dict[str, float] = {}
    best_std: Dict[str, float] = {}
    for key in sorted(best_keys):
        values = [
            float(summary["best"][key])
            for summary in summaries
            if isinstance(summary.get("best", {}).get(key), (int, float))
        ]
        if values:
            best_mean[key] = float(statistics.mean(values))
            best_std[key] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    
    aggregated = {
        "mode": summaries[0].get("mode"),
        "baseline": "lstm",
        "seeds": seeds,
        "per_seed": summaries,
        "best_mean": best_mean,
        "best_std": best_std,
        "config": {k: v for k, v in asdict(cfg).items() if k not in {"config_path", "seed"}},
    }
    return aggregated


def save_summary(summary: Dict, path: Optional[str]) -> None:
    """Save summary to JSON file."""
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("结果已保存到: %s", out_path)


def run_experiment(cfg: TrainConfig) -> Dict:
    """Run LSTM baseline experiment."""
    # Initialize W&B
    run_id = init_wandb(cfg)
    
    # Load data
    if cfg.split_mode == "dir":
        # dir 模式：从预定义的 train/val/test 目录加载数据
        LOGGER.info("Loading data from predefined directories...")
        train_texts, train_labels, train_ids = load_md_reports_from_dir(
            cfg.train_dir, progress=cfg.progress
        )
        val_texts, val_labels, val_ids = load_md_reports_from_dir(
            cfg.val_dir, progress=cfg.progress
        )
        test_texts, test_labels, test_ids = load_md_reports_from_dir(
            cfg.test_dir, progress=cfg.progress
        )
        
        # 合并所有数据
        texts = train_texts + val_texts + test_texts
        labels = train_labels + val_labels + test_labels
        ids = train_ids + val_ids + test_ids
        
        # 保存划分信息用于后续评估
        n_train = len(train_texts)
        n_val = len(val_texts)
        n_test = len(test_texts)
        
        # Build vocabulary from training data only
        LOGGER.info("Building vocabulary from training data...")
        vocab = build_vocab(train_texts, cfg.vocab_size)
    elif cfg.split_mode == "time_ood":
        # time_ood 模式：基于时间切分数据
        from src.utils.time_split import load_manifest_with_time, split_by_time
        
        LOGGER.info("Loading data with time-based split...")
        mal_df, ben_df = load_manifest_with_time(
            cfg.malicious_manifest_path,
            cfg.benign_manifest_path,
            mal_dir=cfg.mal_dir,
            benign_dir=cfg.benign_dir,
        )
        
        train_end_date = getattr(cfg, 'train_end_date', '2025-04-23 08:08:46')
        val_end_date = getattr(cfg, 'val_end_date', '2025-06-14 11:39:39')
        
        (
            train_paths, train_labels_list, train_ids_list,
            val_paths, val_labels_list, val_ids_list,
            test_paths, test_labels_list, test_ids_list,
        ) = split_by_time(
            mal_df,
            ben_df,
            train_end_date,
            val_end_date,
            mal_dir=cfg.mal_dir,
            benign_dir=cfg.benign_dir,
            benign_split_strategy="random",
            random_seed=getattr(cfg, 'seed', 42),
        )
        
        # 加载文本内容
        def load_texts_from_paths(paths):
            texts = []
            for path in paths:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        texts.append(f.read())
                except Exception as e:
                    LOGGER.warning(f"Failed to load {path}: {e}")
                    texts.append("")
            return texts
        
        train_texts = load_texts_from_paths(train_paths)
        val_texts = load_texts_from_paths(val_paths)
        test_texts = load_texts_from_paths(test_paths)
        
        # 合并所有数据
        texts = train_texts + val_texts + test_texts
        labels = train_labels_list + val_labels_list + test_labels_list
        ids = train_ids_list + val_ids_list + test_ids_list
        
        # 保存划分信息用于后续评估
        n_train = len(train_texts)
        n_val = len(val_texts)
        n_test = len(test_texts)
        
        # Build vocabulary from training data only
        LOGGER.info("Building vocabulary from training data...")
        vocab = build_vocab(train_texts, cfg.vocab_size)
    else:
        # 原有的 holdout/cv 模式：使用 mal_dir + benign_dir 加载数据
        mal_dir_path = Path(cfg.mal_dir)
        ben_dir_path = Path(cfg.benign_dir)
        
        mal_files = sorted(mal_dir_path.glob("*.md"))
        ben_files = sorted(ben_dir_path.glob("*.md"))
        
        texts = []
        labels = []
        ids = []
        
        all_files = [(f, 1) for f in mal_files] + [(f, 0) for f in ben_files]
        iterator = all_files
        if cfg.progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Loading reports", total=len(all_files))
        
        for md_file, label in iterator:
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                texts.append(text)
                labels.append(label)
                ids.append(md_file.stem)
            except Exception as e:
                LOGGER.warning(f"Failed to load {md_file}: {e}")
        
        LOGGER.info(f"加载了 {len(texts)} 个报告 (恶意: {len(mal_files)}, 良性: {len(ben_files)})")
        
        # 如果设置了 limit，进行分层抽样
        if cfg.limit and len(texts) > cfg.limit:
            import numpy as np
            from sklearn.model_selection import StratifiedShuffleSplit
            LOGGER.info("限制样本数至 %d，进行分层抽样。", cfg.limit)
            y_arr = np.array(labels, dtype=int)
            idx_all = np.arange(len(labels))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=cfg.limit, random_state=42)
            keep_idx, _ = next(sss.split(idx_all, y_arr))
            texts = [texts[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
            ids = [ids[i] for i in keep_idx]
        
        n_train = n_val = n_test = None
        
        # Build vocabulary from training data
        LOGGER.info("Building vocabulary...")
        train_texts = texts[:len(texts) // 2]  # Use first half for vocab
        vocab = build_vocab(train_texts, cfg.vocab_size)
    
    if not texts:
        raise RuntimeError("未读取到任何样本，请检查 Markdown 目录。")
    
    labels_arr = np.array(labels, dtype=int)
    
    LOGGER.info(f"Vocabulary size: {len(vocab)}")
    LOGGER.info(f"Total samples: {len(texts)}")
    
    # Evaluate
    if cfg.split_mode in {"dir", "time_ood"}:
        summary = evaluate_holdout_with_predefined_indices(
            texts, labels, vocab, n_train, n_val, n_test, cfg
        )
    elif cfg.split_mode == "holdout":
        summary = evaluate_holdout(texts, labels, vocab, cfg)
    else:
        summary = evaluate_cv(texts, labels, vocab, cfg)
    
    # Log to W&B
    if "best" in summary:
        from src.baselines._wandb_common import log_comprehensive_wandb_summary_simple
        
        # Get predictions for visualization
        labels_arr = np.array(labels, dtype=int)
        if cfg.split_mode == "holdout":
            # Re-evaluate to get predictions
            texts_arr = np.array(texts)
            train_idx, val_idx, test_idx = stratified_holdout_indices(labels_arr, cfg.val_ratio, cfg.test_ratio, cfg.seed)
            test_texts = [texts[i] for i in test_idx]
            test_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in test_texts]
            test_dataset = TensorDataset(
                torch.tensor(test_seqs, dtype=torch.long),
                torch.tensor(labels_arr[test_idx], dtype=torch.long),
            )
            test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
            device_obj = torch.device(cfg.device)
            model = LSTMModel(
                vocab_size=len(vocab),
                embedding_dim=cfg.embedding_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ).to(device_obj)
            # Train model
            train_texts_list = [texts[i] for i in train_idx]
            train_seqs = [text_to_sequence(t, vocab, cfg.max_length) for t in train_texts_list]
            train_dataset = TensorDataset(
                torch.tensor(train_seqs, dtype=torch.long),
                torch.tensor(labels_arr[train_idx], dtype=torch.long),
            )
            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
            train_lstm(model, train_loader, cfg.device, cfg.epochs, cfg.learning_rate, False)
            # Get predictions
            test_probs, test_labels = evaluate_lstm(model, test_loader, cfg.device)
            y_true = test_labels
            y_prob = test_probs
        else:
            # CV mode - use summary predictions if available
            y_true = labels_arr
            y_prob = None  # Will be computed from summary if needed
        
        log_comprehensive_wandb_summary_simple(
            summary, labels_arr, cfg, "lstm",
            y_true=y_true if y_true is not None else labels_arr,
            y_prob=y_prob,
        )
        
        if WANDB_AVAILABLE and wandb.run:
            wandb.finish()
            LOGGER.info("实验结果已记录到 W&B")
    
    if cfg.save_raw:
        summary["raw_data"] = {
            "ids": ids,
            "labels": labels_arr.tolist(),
        }
    
    summary["config_digest"] = {
        "md_dir": str(Path(cfg.md_dir).resolve()),
        "baseline": "lstm",
        "limit": cfg.limit,
    }
    
    return summary


# --------------------------------------------------------------------------- #
# 命令行接口
# --------------------------------------------------------------------------- #


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LSTM baseline experiments")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--mal_dir", type=str, default=None, help="Malicious markdown directory (for holdout/cv/time_ood mode)")
    parser.add_argument("--benign_dir", type=str, default=None, help="Benign markdown directory (for holdout/cv/time_ood mode)")
    parser.add_argument("--train_dir", type=str, default=None, help="Training directory (for dir mode)")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation directory (for dir mode)")
    parser.add_argument("--test_dir", type=str, default=None, help="Test directory (for dir mode)")
    parser.add_argument("--split_mode", type=str, default="holdout", choices=["holdout", "cv", "dir", "time_ood"])
    parser.add_argument("--malicious_manifest_path", type=str, default=None, help="Path to malicious manifest CSV (for time_ood mode)")
    parser.add_argument("--benign_manifest_path", type=str, default=None, help="Path to benign manifest CSV (for time_ood mode)")
    parser.add_argument("--train_end_date", type=str, default=None, help="Train end date for time_ood mode")
    parser.add_argument("--val_end_date", type=str, default=None, help="Val end date for time_ood mode")
    parser.add_argument("--limit", type=int, help="Limit samples")
    parser.add_argument("--out", type=str, help="Output JSON file")
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config_file(args.config) if args.config else {}
    
    # Create config
    cfg = TrainConfig(
        md_dir=args.md_dir or config_dict.get("md_dir", ""),
        split_mode=args.split_mode or config_dict.get("split_mode", "holdout"),
        limit=args.limit or config_dict.get("limit"),
        out=args.out or config_dict.get("out"),
        progress=args.progress or config_dict.get("progress", False),
        **{k: v for k, v in config_dict.items() if k not in {
            "md_dir", "split_mode", "limit", "out", "progress"
        }},
    )
    
    cfg.validate()
    
    # Run experiment
    summary = run_experiment(cfg)
    
    # Save results
    save_summary(summary, cfg.out)
    
    # Print summary
    if "best" in summary:
        best = summary["best"]
        LOGGER.info("=" * 70)
        LOGGER.info("实验结果")
        LOGGER.info("=" * 70)
        LOGGER.info("Baseline: LSTM")
        LOGGER.info("Mode: %s", cfg.split_mode)
        LOGGER.info("AUC: %.4f", best.get("auroc", 0))
        LOGGER.info("F1: %.4f", best.get("macro_f1", 0))
        LOGGER.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()

