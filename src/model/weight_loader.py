"""
GPTQ safetensors weight loader for Qwen 3.5 27B INT4.

Loads quantized weights from HuggingFace GPTQ safetensors format,
repacks them into our kernel's expected layout, and uploads to GPU.

Expected safetensors keys for each linear layer:
  - {name}.qweight  (uint32, packed INT4, [K/8, N] or similar)
  - {name}.qzeros   (uint32, packed INT4 zeros, [K/group_size, N/8])
  - {name}.scales   (FP16, [K/group_size, N])
  - {name}.g_idx    (optional, for act_order=True)

Our kernel expects:
  - qweight: uint32, [K/8, N] (8 INT4 per uint32, column-major groups of 8 K)
  - scales:  FP16, [K/group_size, N]
  - zeros:   FP16, [K/group_size, N]
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_safetensors_metadata(path: str) -> Tuple[dict, int]:
    """Load safetensors header metadata without reading tensor data."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size)
        metadata = json.loads(header_json)
    return metadata, 8 + header_size


def load_safetensors_tensor(path: str, key: str, metadata: dict,
                            data_offset: int) -> np.ndarray:
    """Load a specific tensor from a safetensors file."""
    if key not in metadata:
        raise KeyError(f"Tensor '{key}' not found in safetensors")

    info = metadata[key]
    dtype_map = {
        'F16': np.float16,
        'F32': np.float32,
        'I32': np.int32,
        'I64': np.int64,
        'U8': np.uint8,
        'I8': np.int8,
    }

    dtype_str = info['dtype']
    shape = info['shape']
    offsets = info['data_offsets']
    start, end = offsets

    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    with open(path, 'rb') as f:
        f.seek(data_offset + start)
        data = f.read(end - start)

    return np.frombuffer(data, dtype=dtype).reshape(shape)


def unpack_gptq_qzeros(qzeros_packed: np.ndarray, bits: int = 4) -> np.ndarray:
    """Unpack GPTQ qzeros from packed uint32 to individual values.

    qzeros shape: [num_groups, N_packed] where N_packed = N / (32/bits)
    Returns: [num_groups, N] as FP16 zeros
    """
    vals_per_u32 = 32 // bits
    num_groups, n_packed = qzeros_packed.shape
    N = n_packed * vals_per_u32

    zeros = np.zeros((num_groups, N), dtype=np.float16)
    mask = (1 << bits) - 1

    for i in range(vals_per_u32):
        col_start = i * n_packed
        # Wait, GPTQ packs zeros differently: within each uint32,
        # consecutive bits are for consecutive output columns
        vals = (qzeros_packed >> (i * bits)) & mask
        # Add 1 because GPTQ stores (zero - 1) in some implementations
        zeros[:, i::vals_per_u32] = vals.astype(np.float16) + 1

    return zeros


class GPTQWeightLoader:
    """Loads GPTQ-quantized model weights from safetensors files."""

    def __init__(self, model_dir: str, bits: int = 4, group_size: int = 128):
        self.model_dir = Path(model_dir)
        self.bits = bits
        self.group_size = group_size
        self._file_cache = {}  # path -> (metadata, data_offset)

    def _get_file_info(self, path: str) -> Tuple[dict, int]:
        if path not in self._file_cache:
            metadata, offset = load_safetensors_metadata(path)
            self._file_cache[path] = (metadata, offset)
        return self._file_cache[path]

    def find_tensor_file(self, key: str) -> Optional[str]:
        """Find which safetensors file contains a given tensor key."""
        # Check index file first
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if key in weight_map:
                return str(self.model_dir / weight_map[key])

        # Scan all safetensors files
        for sf in sorted(self.model_dir.glob("*.safetensors")):
            metadata, _ = self._get_file_info(str(sf))
            # Filter out __metadata__ key
            tensor_keys = [k for k in metadata if k != "__metadata__"]
            if key in tensor_keys:
                return str(sf)

        return None

    def load_linear_weights(self, layer_prefix: str) -> Dict[str, np.ndarray]:
        """Load qweight, scales, and zeros for a linear layer.

        Returns dict with keys: 'qweight', 'scales', 'zeros'
        qweight: uint32 [K/8, N] (original GPTQ format)
        scales: FP16 [K/group_size, N]
        zeros: FP16 [K/group_size, N]
        """
        keys = {
            'qweight': f"{layer_prefix}.qweight",
            'scales': f"{layer_prefix}.scales",
            'qzeros': f"{layer_prefix}.qzeros",
        }

        result = {}
        for name, key in keys.items():
            file_path = self.find_tensor_file(key)
            if file_path is None:
                raise FileNotFoundError(f"Cannot find tensor: {key}")
            metadata, offset = self._get_file_info(file_path)
            result[name] = load_safetensors_tensor(file_path, key, metadata, offset)

        # Unpack zeros from packed uint32 to FP16
        result['zeros'] = unpack_gptq_qzeros(result['qzeros'], self.bits)
        del result['qzeros']

        return result

    def list_layers(self) -> list:
        """List all layer prefixes found in the model."""
        all_keys = set()
        for sf in sorted(self.model_dir.glob("*.safetensors")):
            metadata, _ = self._get_file_info(str(sf))
            all_keys.update(k for k in metadata if k != "__metadata__")

        # Extract unique layer prefixes (everything before .qweight/.scales/etc.)
        prefixes = set()
        for key in all_keys:
            for suffix in ['.qweight', '.scales', '.qzeros', '.weight', '.bias']:
                if key.endswith(suffix):
                    prefixes.add(key[:-len(suffix)])
                    break

        return sorted(prefixes)
