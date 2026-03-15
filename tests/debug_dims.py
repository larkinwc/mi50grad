"""Verify kv dimensions."""
import sys
sys.path.insert(0, '/opt/mi50grad')
from src.model.qwen import load_config_from_json
from src.inference.tp_engine import TPInferenceEngine
from src.model.weight_loader import QwenWeightLoader

MODEL_DIR = '/opt/models/Qwen3.5-27B-GPTQ-Int4'
config = load_config_from_json(MODEL_DIR)
loader = QwenWeightLoader(MODEL_DIR, config)

print(f"config.head_dim={config.head_dim}")
print(f"config.num_key_value_heads={config.num_key_value_heads}")
print(f"config.hidden_size={config.hidden_size}")
print(f"config.num_attention_heads={config.num_attention_heads}")

# Load one engine to check local values
from src.inference.engine import InferenceEngine
e = InferenceEngine(config, device_id=0, max_seq_len=2048, tp_size=4)
print(f"local_num_kv_heads={e.local_num_kv_heads}")
print(f"kv_dim={e.kv_dim}")
print(f"kv_stride={e.local_num_kv_heads * config.head_dim * 2}")
e.cleanup()
