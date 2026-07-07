from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class FeatureExtractorConfig:
    model_dir: str
    device: str = "cpu"
    dtype: Optional[str] = None


class LayerwiseFeatureExtractor:
    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        dtype_map = {
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower()) if isinstance(dtype, str) else dtype

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs = {"output_hidden_states": True, "trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_dir, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.num_model_layers = self.model.config.num_hidden_layers

    @torch.inference_mode()
    def encode_document_layers(
        self,
        text: str,
        max_tokens: int = 1024,
        stride: int = 256,
        until_layer: Optional[int] = None,
    ) -> torch.Tensor:
        if not text:
            layers = self._effective_layers(until_layer)
            return torch.zeros((layers, self.hidden_size))

        model_max = getattr(self.tokenizer, "model_max_length", None)
        if model_max and model_max > 0:
            max_length = min(max_tokens, model_max)
        else:
            max_length = max_tokens

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            stride=stride if stride < max_length else 0,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # 如果chunk太多，限制batch size以避免显存溢出
        num_chunks = input_ids.shape[0]
        max_chunks_per_batch = 1  # 限制每个batch最多处理1个chunk，最大程度减少显存占用
        
        # 限制最大chunks数量，避免处理超长文档时耗时过长
        # 如果文档太长，只处理前N个chunks
        max_total_chunks = 50  # 最多处理50个chunks
        if num_chunks > max_total_chunks:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"文档过长，产生{num_chunks}个chunks，仅处理前{max_total_chunks}个")
            input_ids = input_ids[:max_total_chunks]
            attention_mask = attention_mask[:max_total_chunks]
            num_chunks = max_total_chunks
        
        if num_chunks > max_chunks_per_batch:
            # 分批处理
            all_layer_vectors = []
            for start_idx in range(0, num_chunks, max_chunks_per_batch):
                end_idx = min(start_idx + max_chunks_per_batch, num_chunks)
                batch_input_ids = input_ids[start_idx:end_idx]
                batch_attention_mask = attention_mask[start_idx:end_idx]
                
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                batch_hidden_states = outputs.hidden_states[1:]  # skip embedding layer
                
                if until_layer is not None:
                    batch_hidden_states = batch_hidden_states[:until_layer]
                
                batch_mask = batch_attention_mask.unsqueeze(-1).type_as(batch_hidden_states[0])
                batch_token_counts = batch_mask.sum(dim=1).clamp(min=1.0)
                
                for layer_idx, layer in enumerate(batch_hidden_states):
                    weighted = (layer * batch_mask).sum(dim=1)  # [batch, hidden]
                    if len(all_layer_vectors) <= layer_idx:
                        all_layer_vectors.append([])
                    all_layer_vectors[layer_idx].append((weighted.sum(dim=0), batch_token_counts.sum(dim=0)))
                
                # 清理显存
                del outputs, batch_hidden_states, batch_mask, batch_token_counts
                torch.cuda.empty_cache()
            
            # 聚合所有batch的结果
            layer_vectors = []
            for layer_aggs in all_layer_vectors:
                total_sum = sum(agg[0] for agg in layer_aggs)
                total_count = sum(agg[1] for agg in layer_aggs)
                layer_vectors.append(total_sum / total_count)
        else:
            # 原始逻辑：一次性处理所有chunk
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states[1:]  # skip embedding layer

            if until_layer is not None:
                hidden_states = hidden_states[:until_layer]

            mask = attention_mask.unsqueeze(-1).type_as(hidden_states[0])
            token_counts = mask.sum(dim=1).clamp(min=1.0)

            layer_vectors = []
            for layer in hidden_states:
                weighted = (layer * mask).sum(dim=1)  # [batch, hidden]
                total_sum = weighted.sum(dim=0)
                total_count = token_counts.sum(dim=0)
                layer_vectors.append(total_sum / total_count)

        features = torch.stack(layer_vectors, dim=0)
        return features.detach().cpu()

    def _effective_layers(self, until_layer: Optional[int]) -> int:
        if until_layer is None:
            return self.num_model_layers
        return max(1, min(until_layer, self.num_model_layers))
