# Test Prompts for Speculative Decoding Validation

This directory contains test prompts for validating speculative decoding implementations across 4 domains.

## Overview

The prompt dataset provides standardized test data for measuring speculative decoding acceptance rates and throughput improvements.

**Location:** `src/inference/prompts.py`  
**Data File:** `data/test_prompts.json`

## Categories

### 1. Code Completion (5 prompts)
Python functions, classes, and control flow patterns:
- Recursive functions (Fibonacci, quicksort)
- Class definitions with methods
- Utility functions with type hints
- Async/await patterns

**Expected n-gram acceptance:** ≥50%

### 2. JSON Completion (5 prompts)
Nested structures, API responses, and configuration files:
- User objects with preferences
- API responses with pagination
- Configuration with nested settings
- Log entries with timestamps
- JSON schema definitions

**Expected n-gram acceptance:** ≥45%

### 3. Conversational (5 prompts)
Multi-turn dialogue and Q&A patterns:
- Educational Q&A
- Casual workplace conversations
- Customer support interactions
- Job interviews
- Tutoring sessions

**Expected n-gram acceptance:** ≥40%

### 4. Repetitive Text (5 prompts)
High n-gram match rate patterns:
- Simple sentence structures
- Color sequences
- Days of week
- Number sequences
- Greek letters

**Expected n-gram acceptance:** ≥70%

## Train/Test Split Methodology

The dataset implements a **60%/40% train/test split** for n-gram cache building:

- **Training (60%)**: Used to build the n-gram cache
- **Testing (40%)**: Used to measure acceptance rates

This prevents training-on-test bias and provides realistic acceptance rate measurements.

### Example Usage

```python
from src.inference.prompts import PromptDataset, TrainTestSplitter

# Load dataset
dataset = PromptDataset()
prompts = dataset.get_prompts(category='code')

# Create train/test split
splitter = TrainTestSplitter(train_ratio=0.6)
train_text, test_text = splitter.split_text(prompts[0].text)

# Build n-gram cache from training portion
from src.inference.speculative import NgramCache
cache = NgramCache(n=3)
cache.build_from_sequence([ord(c) % 256 for c in train_text])

# Measure acceptance on test portion
```

## Tokenizer Integration

The module includes a `TokenizerAdapter` for processing prompts with various tokenizers:

```python
from src.inference.prompts import TokenizerAdapter

# Character-level tokenizer (default)
tokenizer = TokenizerAdapter.create_char_level()
tokens = tokenizer.encode("Hello, World!")
text = tokenizer.decode(tokens)

# With custom tokenizer (e.g., HuggingFace)
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
adapter = TokenizerAdapter(hf_tokenizer)
tokens = adapter.encode(prompt_text)
```

## File Structure

```
data/
├── test_prompts.json          # Serialized prompt dataset
└── PROMPTS_README.md          # This file

src/inference/
└── prompts.py                 # Prompt module implementation

tests/
├── verify_prompts.py          # Verification script
└── val_m6_speculative.py      # Validation tests using prompts
```

## JSON Format

The `test_prompts.json` file uses this structure:

```json
{
  "categories": {
    "code": [
      {
        "text": "def fibonacci(n):...",
        "category": "code",
        "description": "Recursive Fibonacci function",
        "expected_pattern": "Function definition",
        "metadata": {
          "complexity": "medium",
          "pattern": "recursion"
        }
      }
    ]
  }
}
```

## Verification

Run the verification script to ensure all requirements are met:

```bash
python3 tests/verify_prompts.py
```

Expected output:
```
✓ ALL REQUIREMENTS MET - m2-prepare-real-text-prompts COMPLETE
```

## Integration with Validation Tests

The prompts are automatically loaded in `tests/val_m6_speculative.py`:

```python
from src.inference.prompts import PromptDataset, PromptCategory

dataset = PromptDataset()
TEST_PROMPTS = {
    'code': [p.text for p in dataset.get_prompts(PromptCategory.CODE)],
    'json': [p.text for p in dataset.get_prompts(PromptCategory.JSON)],
    # ... etc
}
```

## Adding Custom Prompts

To add custom prompts to the dataset:

```python
from src.inference.prompts import PromptDataset, Prompt, PromptCategory

dataset = PromptDataset()
custom_prompt = Prompt(
    text="Your custom prompt text here",
    category=PromptCategory.CODE,
    description="Description of the prompt",
    expected_pattern="Expected continuation pattern",
    metadata={"complexity": "high", "pattern": "custom"}
)
dataset.add_prompt(custom_prompt)
dataset.save_to_file('data/custom_prompts.json')
```

## Performance Expectations

Based on the prompt characteristics:

| Category | N-gram Acceptance | EAGLE Acceptance | Notes |
|----------|------------------|------------------|-------|
| Code | ~50-60% | ~40-50% | Structured patterns |
| JSON | ~45-55% | ~35-45% | Repetitive syntax |
| Conversational | ~40-50% | ~30-40% | Less predictable |
| Repetitive | ~70-80% | ~50-60% | High pattern match |

These estimates are based on n-gram size 3 and max draft length 5.

## References

- Feature: `m2-prepare-real-text-prompts`
- Milestone: `m6-speculative-validation`
- Validation Contract: `VAL-M2-001` through `VAL-M2-008`
