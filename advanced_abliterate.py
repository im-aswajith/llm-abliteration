import os
import gc
import json
import logging
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize_scalar

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AdvancedAbliterationConfig:
    model_id: str = "unsloth/Llama-3.2-3B-Instruct"
    output_dir: str = "./Llama-3.2-3B-Instruct-advanced-abliterated"

  # replace with you're own dataset or questions
    harmful_prompts: List[str] = field(default_factory=lambda: [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ])
    
    harmless_prompts: List[str] = field(default_factory=lambda: [
        "Tell me about the history of computers",
        "Explain how photosynthesis works",
        "What are the benefits of exercise?",
        "How do I bake a chocolate cake?",
        "Explain the water cycle",
        "What is the capital of France?",
        "Tell me about renewable energy",
        "How do I plant a garden?",
        "Explain the theory of relativity",
        "What are the main causes of climate change?",
        "How does a refrigerator work?",
        "Tell me about the Roman Empire",
        "What is the Pythagorean theorem?",
        "How do vaccines work?",
        "Explain the process of photosynthesis",
        "What are the symptoms of the common cold?",
    ])
    
    num_samples: int = 64
    batch_size: int = 4
    max_length: int = 512
    layers: Optional[List[int]] = None
    alpha: float = 1.0
    
    use_norm_preserving: bool = True
    use_biprojection: bool = True
    use_triangular_falloff: bool = True
    use_multi_direction: bool = True
    use_auto_optimization: bool = True
    use_lora_assist: bool = False
    
    load_in_4bit: bool = False
    torch_dtype: torch.dtype = torch.bfloat16


class AdvancedLlamaAbliterator:
    
    def __init__(self, config: AdvancedAbliterationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.refusal_directions = {}
        self.harmless_directions = {}
        self.optimal_alphas = {}
        
    def load_model(self) -> None:
        logger.info(f"Loading model: {self.config.model_id}")
        
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _prepare_messages(self, prompts: List[str]) -> List[List[Dict]]:
        return [[{"role": "user", "content": p}] for p in prompts]
    
    def _get_hidden_states(
        self,
        prompts: List[str],
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        if layer_indices is None:
            num_layers = self.model.config.num_hidden_layers
            layer_indices = list(range(num_layers))
        
        messages = self._prepare_messages(prompts)
        texts = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)
        
        hidden_states_by_layer = {i: [] for i in layer_indices}
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            for i, layer_idx in enumerate(layer_indices):
                hs = outputs.hidden_states[layer_idx + 1]
                last_token_hs = hs[:, -1, :]
                hidden_states_by_layer[layer_idx].append(last_token_hs)
        
        for layer_idx in layer_indices:
            hidden_states_by_layer[layer_idx] = torch.cat(
                hidden_states_by_layer[layer_idx], dim=0
            )
        
        return hidden_states_by_layer
    
    def compute_multi_direction_refusal(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str]
    ) -> None:
        logger.info("Computing multi-direction refusal vectors...")
        
        n = min(self.config.num_samples, len(harmful_prompts), len(harmless_prompts))
        harmful_prompts = harmful_prompts[:n]
        harmless_prompts = harmless_prompts[:n]
        
        num_layers = self.model.config.num_hidden_layers
        layer_indices = self.config.layers or list(range(num_layers))
        
        all_harmful_hidden = {}
        all_harmless_hidden = {}
        
        for i in tqdm(range(0, n, self.config.batch_size), desc="Processing batches"):
            batch_harmful = harmful_prompts[i:i + self.config.batch_size]
            batch_harmless = harmless_prompts[i:i + self.config.batch_size]
            
            hs_harmful = self._get_hidden_states(batch_harmful, layer_indices)
            hs_harmless = self._get_hidden_states(batch_harmless, layer_indices)
            
            for layer_idx in layer_indices:
                if layer_idx not in all_harmful_hidden:
                    all_harmful_hidden[layer_idx] = []
                    all_harmless_hidden[layer_idx] = []
                all_harmful_hidden[layer_idx].append(hs_harmful[layer_idx])
                all_harmless_hidden[layer_idx].append(hs_harmless[layer_idx])
        
        for layer_idx in layer_indices:
            harmful_hs = torch.cat(all_harmful_hidden[layer_idx], dim=0)
            harmless_hs = torch.cat(all_harmless_hidden[layer_idx], dim=0)
            
            harmful_mean = harmful_hs.mean(dim=0, keepdim=True)
            harmless_mean = harmless_hs.mean(dim=0, keepdim=True)
            
            main_refusal = harmful_mean - harmless_mean
            main_refusal = main_refusal / (main_refusal.norm() + 1e-8)
            
            self.refusal_directions[layer_idx] = [main_refusal]
            self.harmless_directions[layer_idx] = harmless_mean / (harmless_mean.norm() + 1e-8)
            
            if self.config.use_multi_direction:
                centered = harmful_hs - harmful_mean
                # Cast to float32 for numerical stability and to support eigh on CUDA
                centered_f32 = centered.float()
                cov = (centered_f32.T @ centered_f32) / (centered_f32.size(0) - 1)
                
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                    # Convert eigenvectors back to original dtype
                    eigenvectors = eigenvectors.to(harmful_hs.dtype)
                    
                    k = min(3, len(eigenvalues))
                    top_indices = torch.topk(eigenvalues.abs(), k).indices
                    
                    for idx in top_indices:
                        extra_dir = eigenvectors[:, idx].unsqueeze(0)
                        extra_dir = extra_dir - (extra_dir @ main_refusal.T) * main_refusal
                        if extra_dir.norm() > 1e-6:
                            extra_dir = extra_dir / (extra_dir.norm() + 1e-8)
                            self.refusal_directions[layer_idx].append(extra_dir)
                except Exception as e:
                    logger.warning(f"Failed to compute extra directions for layer {layer_idx}: {e}")
            
            logger.info(f"Layer {layer_idx}: {len(self.refusal_directions[layer_idx])} refusal directions")
    
    def optimize_alpha_per_layer(self) -> None:
        if not self.config.use_auto_optimization:
            for layer_idx in self.refusal_directions:
                self.optimal_alphas[layer_idx] = self.config.alpha
            return
        
        logger.info("Optimizing alpha per layer...")
        num_layers = self.model.config.num_hidden_layers
        
        for layer_idx in self.refusal_directions:
            direction_strength = 0.0
            for direction in self.refusal_directions[layer_idx]:
                direction_strength += direction.norm().item()
            direction_strength /= len(self.refusal_directions[layer_idx])
            
            base_alpha = self.config.alpha
            
            if self.config.use_triangular_falloff:
                normalized_pos = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
                falloff = 1.0 - 2.0 * abs(normalized_pos - 0.5)
                falloff = max(0.3, falloff)
            else:
                falloff = 1.0
            
            strength_factor = min(1.5, max(0.5, direction_strength * 2.0))
            self.optimal_alphas[layer_idx] = base_alpha * falloff * strength_factor
            logger.debug(f"Layer {layer_idx}: alpha = {self.optimal_alphas[layer_idx]:.4f}")
    
    def _orthogonalize_matrix_advanced(
        self,
        weight: torch.Tensor,
        directions: List[torch.Tensor],
        harmless_direction: Optional[torch.Tensor] = None,
        alpha: float = 1.0
    ) -> torch.Tensor:
        if weight.dim() != 2:
            logger.warning(f"Non-2D weight tensor {weight.shape}, skipping")
            return weight
        
        all_dirs = []
        for d in directions:
            d = d.squeeze(0) if d.dim() == 2 and d.size(0) == 1 else d
            
            if d.size(0) != weight.size(1):
                if d.size(0) < weight.size(1):
                    d = F.pad(d, (0, weight.size(1) - d.size(0)))
                else:
                    d = d[:weight.size(1)]
            
            d = d / (d.norm() + 1e-8)
            all_dirs.append(d)
        
        if not all_dirs:
            return weight
        
        if self.config.use_biprojection and harmless_direction is not None:
            hd = harmless_direction.squeeze(0) if harmless_direction.dim() == 2 else harmless_direction
            if hd.size(0) != weight.size(1):
                if hd.size(0) < weight.size(1):
                    hd = F.pad(hd, (0, weight.size(1) - hd.size(0)))
                else:
                    hd = hd[:weight.size(1)]
            hd = hd / (hd.norm() + 1e-8)
            
            for i in range(len(all_dirs)):
                proj_on_harmless = (all_dirs[i] @ hd) * hd
                all_dirs[i] = all_dirs[i] - proj_on_harmless
                if all_dirs[i].norm() > 1e-8:
                    all_dirs[i] = all_dirs[i] / (all_dirs[i].norm() + 1e-8)
        
        ortho_dirs = []
        for d in all_dirs:
            for existing in ortho_dirs:
                d = d - (d @ existing) * existing
            if d.norm() > 1e-8:
                d = d / (d.norm() + 1e-8)
                ortho_dirs.append(d)
        
        if not ortho_dirs:
            return weight
        
        if self.config.use_norm_preserving:
            weight_norm = weight.norm(dim=1, keepdim=True)
            weight_dir = weight / (weight_norm + 1e-8)
            
            for d in ortho_dirs:
                proj_on_dir = (weight_dir @ d).unsqueeze(1)
                weight_dir = weight_dir - alpha * proj_on_dir * d.unsqueeze(0)
            
            weight_dir = weight_dir / (weight_dir.norm(dim=1, keepdim=True) + 1e-8)
            new_weight = weight_dir * weight_norm
        else:
            new_weight = weight.clone()
            for d in ortho_dirs:
                proj = torch.outer(
                    new_weight @ d / (d.norm()**2 + 1e-8),
                    d
                )
                new_weight = new_weight - alpha * proj
        
        return new_weight
    
    def apply_advanced_abliteration(self) -> None:
        logger.info("Applying advanced abliteration to model weights...")
        
        if not self.refusal_directions:
            raise ValueError("No refusal directions computed.")
        
        self.optimize_alpha_per_layer()
        
        num_layers = self.model.config.num_hidden_layers
        modified_count = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'weight' not in name or 'layernorm' in name.lower() or 'norm' in name.lower():
                continue
            
            layer_idx = None
            for i in range(num_layers):
                if f'.{i}.' in name or f'layers.{i}.' in name:
                    layer_idx = i
                    break
            
            if layer_idx is None or layer_idx not in self.refusal_directions:
                continue
            
            directions = self.refusal_directions[layer_idx]
            harmless_dir = self.harmless_directions.get(layer_idx)
            alpha = self.optimal_alphas.get(layer_idx, self.config.alpha)
            
            try:
                with torch.no_grad():
                    new_weight = self._orthogonalize_matrix_advanced(
                        param.data,
                        directions,
                        harmless_dir,
                        alpha
                    )
                    param.data.copy_(new_weight)
                    modified_count += 1
            except Exception as e:
                logger.warning(f"Failed to modify {name}: {e}")
        
        logger.info(f"Applied advanced abliteration to {modified_count} weight matrices")
    
    def save_model(self) -> None:
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving abliterated model to {output_path}")
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        config_path = output_path / "advanced_abliteration_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "base_model": self.config.model_id,
                "num_samples": self.config.num_samples,
                "alpha": self.config.alpha,
                "use_norm_preserving": self.config.use_norm_preserving,
                "use_biprojection": self.config.use_biprojection,
                "use_triangular_falloff": self.config.use_triangular_falloff,
                "use_multi_direction": self.config.use_multi_direction,
                "use_auto_optimization": self.config.use_auto_optimization,
                "layers": self.config.layers,
                "num_refusal_directions": {
                    str(k): len(v) for k, v in self.refusal_directions.items()
                }
            }, f, indent=2)
        
        logger.info(f"Model saved successfully to {output_path}")
    
    def run(self) -> None:
        try:
            self.load_model()
            self.compute_multi_direction_refusal(
                self.config.harmful_prompts,
                self.config.harmless_prompts
            )
            self.apply_advanced_abliteration()
            self.save_model()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Abliteration for Llama-3.2-3B-Instruct"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-3B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Llama-3.2-3B-Instruct-advanced-abliterated",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples per category"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Base ablation strength"
    )
    parser.add_argument(
        "--no-norm-preserving",
        action="store_true",
        help="Disable norm-preserving"
    )
    parser.add_argument(
        "--no-biprojection",
        action="store_true",
        help="Disable biprojection"
    )
    parser.add_argument(
        "--no-triangular-falloff",
        action="store_true",
        help="Disable triangular falloff"
    )
    parser.add_argument(
        "--no-multi-direction",
        action="store_true",
        help="Disable multi-direction"
    )
    parser.add_argument(
        "--no-auto-optimization",
        action="store_true",
        help="Disable auto-optimization"
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = AdvancedAbliterationConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        alpha=args.alpha,
        use_norm_preserving=not args.no_norm_preserving,
        use_biprojection=not args.no_biprojection,
        use_triangular_falloff=not args.no_triangular_falloff,
        use_multi_direction=not args.no_multi_direction,
        use_auto_optimization=not args.no_auto_optimization,
        load_in_4bit=args.load_4bit,
    )
    
    abliterator = AdvancedLlamaAbliterator(config)
    abliterator.run()


if __name__ == "__main__":
    main()
