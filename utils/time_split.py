"""
Time-based Out-of-Distribution (Time-OOD) Data Splitting Utilities

This module provides functions for splitting datasets based on temporal information,
ensuring that training data comes from earlier time periods than validation and test data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

LOGGER = logging.getLogger(__name__)


def load_manifest_with_time(
    malicious_manifest_path: str,
    benign_manifest_path: str,
    mal_dir: Optional[str] = None,
    benign_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load manifest CSVs and merge with file paths.
    
    Args:
        malicious_manifest_path: Path to malicious manifest CSV
        benign_manifest_path: Path to benign manifest CSV
        mal_dir: Directory containing malicious markdown files (optional)
        benign_dir: Directory containing benign markdown files (optional)
    
    Returns:
        Tuple of (malicious_df, benign_df) with file paths added
    """
    mal_df = pd.read_csv(malicious_manifest_path)
    ben_df = pd.read_csv(benign_manifest_path)
    
    # Parse time column for malicious data
    if 'first_seen' in mal_df.columns:
        mal_df['first_seen'] = pd.to_datetime(mal_df['first_seen'])
        mal_df = mal_df.sort_values('first_seen').reset_index(drop=True)
    else:
        raise ValueError("malicious manifest must contain 'first_seen' column")
    
    # Add file paths if directories provided
    if mal_dir:
        mal_dir_path = Path(mal_dir)
        mal_df['file_path'] = mal_df['sha256'].apply(
            lambda x: mal_dir_path / f"{x}.md"
        )
    
    if benign_dir:
        ben_dir_path = Path(benign_dir)
        ben_df['file_path'] = ben_df['sha256'].apply(
            lambda x: ben_dir_path / f"{x}.md"
        )
    
    return mal_df, ben_df


def split_by_time(
    malicious_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    mal_dir: Optional[str] = None,
    benign_dir: Optional[str] = None,
    benign_split_strategy: str = "random",
    random_seed: int = 42,
) -> Tuple[
    List[Path], List[int], List[str],  # train
    List[Path], List[int], List[str],  # val
    List[Path], List[int], List[str],  # test
]:
    """Split data by time for Time-OOD experiments.
    
    Args:
        malicious_df: DataFrame with malicious samples and 'first_seen' column
        benign_df: DataFrame with benign samples
        train_end_date: End date for training set (format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
        val_end_date: End date for validation set (format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
        mal_dir: Directory containing malicious markdown files
        benign_dir: Directory containing benign markdown files
        benign_split_strategy: How to split benign data ("random" or "match_ratio")
        random_seed: Random seed for benign data splitting
    
    Returns:
        Tuple of (train_paths, train_labels, train_ids, val_paths, val_labels, val_ids,
                 test_paths, test_labels, test_ids)
    """
    # Parse dates
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    
    if train_end >= val_end:
        raise ValueError("train_end_date must be before val_end_date")
    
    # Split malicious data by time
    mal_train = malicious_df[malicious_df['first_seen'] <= train_end].copy()
    mal_val = malicious_df[
        (malicious_df['first_seen'] > train_end) & 
        (malicious_df['first_seen'] <= val_end)
    ].copy()
    mal_test = malicious_df[malicious_df['first_seen'] > val_end].copy()
    
    LOGGER.info(
        f"Malicious data split: Train={len(mal_train)}, Val={len(mal_val)}, Test={len(mal_test)}"
    )
    
    # Split benign data
    if benign_split_strategy == "random":
        # Random stratified split to match malicious proportions
        benign_labels = np.zeros(len(benign_df), dtype=int)
        idx = np.arange(len(benign_df))
        
        # Calculate target sizes based on malicious split
        total_mal = len(malicious_df)
        train_ratio = len(mal_train) / total_mal
        val_ratio = len(mal_val) / total_mal
        test_ratio = len(mal_test) / total_mal
        
        # First split: train+val vs test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        train_val_idx, test_idx = next(sss1.split(idx, benign_labels))
        
        # Second split: train vs val
        val_size_rel = val_ratio / (1 - test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=random_seed + 1)
        train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, benign_labels[train_val_idx]))
        
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        
        ben_train = benign_df.iloc[train_idx].copy()
        ben_val = benign_df.iloc[val_idx].copy()
        ben_test = benign_df.iloc[test_idx].copy()
        
    elif benign_split_strategy == "match_ratio":
        # Match exact ratios from malicious split
        total_mal = len(malicious_df)
        train_ratio = len(mal_train) / total_mal
        val_ratio = len(mal_val) / total_mal
        test_ratio = len(mal_test) / total_mal
        
        total_ben = len(benign_df)
        train_size = int(total_ben * train_ratio)
        val_size = int(total_ben * val_ratio)
        test_size = total_ben - train_size - val_size
        
        benign_labels = np.zeros(len(benign_df), dtype=int)
        idx = np.arange(len(benign_df))
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=random_seed)
        train_idx, rest_idx = next(sss.split(idx, benign_labels))
        
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=random_seed + 1)
        val_idx_rel, test_idx_rel = next(sss2.split(rest_idx, benign_labels[rest_idx]))
        val_idx = rest_idx[val_idx_rel]
        test_idx = rest_idx[test_idx_rel]
        
        ben_train = benign_df.iloc[train_idx].copy()
        ben_val = benign_df.iloc[val_idx].copy()
        ben_test = benign_df.iloc[test_idx].copy()
    else:
        raise ValueError(f"Unknown benign_split_strategy: {benign_split_strategy}")
    
    LOGGER.info(
        f"Benign data split: Train={len(ben_train)}, Val={len(ben_val)}, Test={len(ben_test)}"
    )
    
    # Combine malicious and benign data
    def prepare_split(mal_subset, ben_subset, split_name):
        paths = []
        labels = []
        ids = []
        
        # Add malicious samples
        for _, row in mal_subset.iterrows():
            if mal_dir:
                paths.append(Path(mal_dir) / f"{row['sha256']}.md")
            elif 'file_path' in row:
                paths.append(Path(row['file_path']))
            else:
                paths.append(None)
            labels.append(1)
            ids.append(row['sha256'])
        
        # Add benign samples
        for _, row in ben_subset.iterrows():
            if benign_dir:
                paths.append(Path(benign_dir) / f"{row['sha256']}.md")
            elif 'file_path' in row:
                paths.append(Path(row['file_path']))
            else:
                paths.append(None)
            labels.append(0)
            ids.append(row['sha256'])
        
        # Filter out None paths
        valid_indices = [i for i, p in enumerate(paths) if p is not None]
        paths = [paths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        
        LOGGER.info(f"{split_name} split: {len(paths)} samples ({sum(labels)} malicious, {len(labels)-sum(labels)} benign)")
        
        return paths, labels, ids
    
    train_paths, train_labels, train_ids = prepare_split(mal_train, ben_train, "Train")
    val_paths, val_labels, val_ids = prepare_split(mal_val, ben_val, "Val")
    test_paths, test_labels, test_ids = prepare_split(mal_test, ben_test, "Test")
    
    return (
        train_paths, train_labels, train_ids,
        val_paths, val_labels, val_ids,
        test_paths, test_labels, test_ids,
    )


def get_time_split_indices(
    malicious_df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get indices for time-based split (for malicious data only).
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    
    train_mask = malicious_df['first_seen'] <= train_end
    val_mask = (malicious_df['first_seen'] > train_end) & (malicious_df['first_seen'] <= val_end)
    test_mask = malicious_df['first_seen'] > val_end
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return train_indices, val_indices, test_indices

