#!/usr/bin/env python3
"""
src/inference/prompts.py — Test Prompts for Speculative Decoding Validation.

Provides test prompts across 4 domains for validating speculative decoding:
  - Code completion (Python functions, classes)
  - JSON completion (nested structures, key-value patterns)
  - Conversational (multi-turn dialogue)
  - Repetitive text (high n-gram match rate)

Features:
  - Train/test split methodology (60%/40%) for n-gram cache building
  - Tokenizer integration for processing prompts
  - Multiple prompts per domain for robust testing
  - Easy loading and iteration utilities

USAGE:
    from src.inference.prompts import PromptDataset, TrainTestSplitter
    
    # Load all prompts
    dataset = PromptDataset()
    prompts = dataset.get_prompts(category='code')
    
    # Create train/test split
    splitter = TrainTestSplitter(train_ratio=0.6)
    train_ids, test_ids = splitter.split(prompt_token_ids)
    
    # Load from file
    dataset = PromptDataset.from_file('data/test_prompts.json')
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PromptCategory(str, Enum):
    """Categories of test prompts for speculative decoding validation."""
    CODE = "code"
    JSON = "json"
    CONVERSATIONAL = "conversational"
    REPETITIVE = "repetitive"


@dataclass
class Prompt:
    """A single test prompt with metadata."""
    text: str
    category: PromptCategory
    description: str = ""
    expected_pattern: str = ""  # Description of expected continuation pattern
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "category": self.category.value,
            "description": self.description,
            "expected_pattern": self.expected_pattern,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        """Create Prompt from dictionary."""
        return cls(
            text=data["text"],
            category=PromptCategory(data["category"]),
            description=data.get("description", ""),
            expected_pattern=data.get("expected_pattern", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class TrainTestSplit:
    """Train/test split result for a prompt."""
    train_text: str
    test_text: str
    train_token_ids: List[int]
    test_token_ids: List[int]
    split_ratio: float  # e.g., 0.6 for 60/40 split
    
    def __repr__(self) -> str:
        return f"TrainTestSplit(train_len={len(self.train_text)}, test_len={len(self.test_text)}, ratio={self.split_ratio:.0%})"


class PromptDataset:
    """
    Dataset of test prompts for speculative decoding validation.
    
    Provides prompts across 4 domains:
      - Code: Python functions, classes, control flow
      - JSON: Nested structures, API responses, configuration
      - Conversational: Multi-turn dialogue, Q&A
      - Repetitive: High n-gram match rate patterns
    
    Usage:
        dataset = PromptDataset()
        code_prompts = dataset.get_prompts(PromptCategory.CODE)
        all_prompts = dataset.get_all_prompts()
    """
    
    def __init__(self):
        """Initialize with default test prompts."""
        self._prompts: Dict[PromptCategory, List[Prompt]] = {
            cat: [] for cat in PromptCategory
        }
        self._initialize_default_prompts()
    
    def _initialize_default_prompts(self):
        """Initialize with comprehensive test prompts for all categories."""
        
        # ====================================================================
        # CODE COMPLETION PROMPTS
        # ====================================================================
        code_prompts = [
            Prompt(
                text="def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nfor i in range(10):\n    print(fibonacci(i))\n",
                category=PromptCategory.CODE,
                description="Recursive Fibonacci function with loop",
                expected_pattern="Function definition and iterative usage",
                metadata={"complexity": "medium", "pattern": "recursion"}
            ),
            Prompt(
                text="class DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    \n    def process(self):\n        result = []\n        for item in self.data:\n            if item > 0:\n                result.append(item * 2)\n        return result\n",
                category=PromptCategory.CODE,
                description="Class definition with method",
                expected_pattern="Class with initialization and processing method",
                metadata={"complexity": "medium", "pattern": "oop"}
            ),
            Prompt(
                text="def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
                category=PromptCategory.CODE,
                description="Quicksort algorithm implementation",
                expected_pattern="Divide and conquer sorting algorithm",
                metadata={"complexity": "high", "pattern": "recursion"}
            ),
            Prompt(
                text="import json\nfrom typing import Dict, List\n\ndef load_config(path: str) -> Dict:\n    with open(path, 'r') as f:\n        return json.load(f)\n\ndef save_config(config: Dict, path: str):\n    with open(path, 'w') as f:\n        json.dump(config, f, indent=2)\n",
                category=PromptCategory.CODE,
                description="Utility functions with type hints",
                expected_pattern="File I/O operations with JSON",
                metadata={"complexity": "low", "pattern": "io"}
            ),
            Prompt(
                text="async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            if response.status == 200:\n                return await response.json()\n            else:\n                raise Exception(f\"Error: {response.status}\")\n",
                category=PromptCategory.CODE,
                description="Async HTTP request handler",
                expected_pattern="Async/await pattern with error handling",
                metadata={"complexity": "medium", "pattern": "async"}
            ),
        ]
        
        # ====================================================================
        # JSON COMPLETION PROMPTS
        # ====================================================================
        json_prompts = [
            Prompt(
                text='{"user": {"id": 12345, "name": "John Doe", "email": "john@example.com", "preferences": {"theme": "dark", "notifications": true}}, "metadata": {"created_at": "2024-01-15", "updated_at": "2024-01-20"}}',
                category=PromptCategory.JSON,
                description="Nested user object with preferences",
                expected_pattern="User profile with nested preferences and metadata",
                metadata={"complexity": "medium", "pattern": "nested_objects"}
            ),
            Prompt(
                text='{"api_response": {"status": "success", "data": [{"product_id": "P001", "name": "Widget", "price": 29.99, "in_stock": true}, {"product_id": "P002", "name": "Gadget", "price": 49.99, "in_stock": false}], "pagination": {"page": 1, "per_page": 10, "total": 2}}}',
                category=PromptCategory.JSON,
                description="API response with product list",
                expected_pattern="E-commerce API response with pagination",
                metadata={"complexity": "high", "pattern": "arrays"}
            ),
            Prompt(
                text='{"config": {"database": {"host": "localhost", "port": 5432, "name": "mydb", "credentials": {"username": "admin", "password": "secret"}}, "cache": {"enabled": true, "ttl": 3600, "backend": "redis"}}}',
                category=PromptCategory.JSON,
                description="Configuration file with nested settings",
                expected_pattern="Application configuration with database and cache",
                metadata={"complexity": "medium", "pattern": "config"}
            ),
            Prompt(
                text='{"logs": [{"timestamp": "2024-01-20T10:30:00Z", "level": "INFO", "message": "Server started"}, {"timestamp": "2024-01-20T10:31:00Z", "level": "WARNING", "message": "High memory usage"}, {"timestamp": "2024-01-20T10:32:00Z", "level": "ERROR", "message": "Connection timeout"}]}',
                category=PromptCategory.JSON,
                description="Log entries array with timestamps",
                expected_pattern="Chronological log entries with levels",
                metadata={"complexity": "medium", "pattern": "time_series"}
            ),
            Prompt(
                text='{"schema": {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "name"]}}',
                category=PromptCategory.JSON,
                description="JSON schema definition",
                expected_pattern="Schema definition with types and constraints",
                metadata={"complexity": "high", "pattern": "schema"}
            ),
        ]
        
        # ====================================================================
        # CONVERSATIONAL PROMPTS
        # ====================================================================
        conversational_prompts = [
            Prompt(
                text="User: Can you help me understand how machine learning works?\nAssistant: Of course! Machine learning is a subset of artificial intelligence that enables computers to learn from data.\nUser: What are the main types?\nAssistant:",
                category=PromptCategory.CONVERSATIONAL,
                description="Educational Q&A about ML",
                expected_pattern="Multi-turn educational dialogue",
                metadata={"complexity": "medium", "pattern": "qa"}
            ),
            Prompt(
                text="Person A: Hey, did you finish the report?\nPerson B: Almost done! Just need to add the conclusion.\nPerson A: Great! When do you think you'll submit it?\nPerson B:",
                category=PromptCategory.CONVERSATIONAL,
                description="Casual workplace conversation",
                expected_pattern="Informal dialogue about work progress",
                metadata={"complexity": "low", "pattern": "casual"}
            ),
            Prompt(
                text="Customer: I'd like to return this item please.\nSupport: Of course! Can I have your order number?\nCustomer: It's ORD-12345.\nSupport: Thank you. What's the reason for the return?\nCustomer:",
                category=PromptCategory.CONVERSATIONAL,
                description="Customer support interaction",
                expected_pattern="Service-oriented dialogue",
                metadata={"complexity": "medium", "pattern": "support"}
            ),
            Prompt(
                text="Interviewer: Tell me about your experience with Python.\nCandidate: I've been using Python for 5 years, mainly for data science projects.\nInterviewer: What libraries are you most comfortable with?\nCandidate:",
                category=PromptCategory.CONVERSATIONAL,
                description="Job interview dialogue",
                expected_pattern="Professional interview exchange",
                metadata={"complexity": "medium", "pattern": "interview"}
            ),
            Prompt(
                text="Student: I don't understand this calculus problem.\nTutor: Let's break it down step by step. What's the first thing you notice?\nStudent: The function has both x and y variables.\nTutor: Good observation! That means we need to use",
                category=PromptCategory.CONVERSATIONAL,
                description="Tutoring session dialogue",
                expected_pattern="Educational guidance conversation",
                metadata={"complexity": "medium", "pattern": "tutoring"}
            ),
        ]
        
        # ====================================================================
        # REPETITIVE TEXT PROMPTS
        # ====================================================================
        repetitive_prompts = [
            Prompt(
                text="the cat sat on the mat. the dog ran in the park. the bird flew in the sky. the fish swam in the water. the rabbit hopped in the garden.",
                category=PromptCategory.REPETITIVE,
                description="Simple repetitive sentence structure",
                expected_pattern="Subject-verb-location pattern repetition",
                metadata={"complexity": "low", "pattern": "simple_repetition"}
            ),
            Prompt(
                text="red blue green yellow red blue green yellow red blue green yellow red blue green yellow",
                category=PromptCategory.REPETITIVE,
                description="Color sequence repetition",
                expected_pattern="Four-color cycle repetition",
                metadata={"complexity": "low", "pattern": "color_cycle"}
            ),
            Prompt(
                text="Monday Tuesday Wednesday Thursday Friday Saturday Sunday Monday Tuesday Wednesday Thursday Friday Saturday Sunday",
                category=PromptCategory.REPETITIVE,
                description="Days of week repetition",
                expected_pattern="Seven-day cycle repetition",
                metadata={"complexity": "low", "pattern": "day_cycle"}
            ),
            Prompt(
                text="one two three four five six seven eight nine ten one two three four five six seven eight nine ten",
                category=PromptCategory.REPETITIVE,
                description="Number sequence repetition",
                expected_pattern="Ten-number cycle repetition",
                metadata={"complexity": "low", "pattern": "number_cycle"}
            ),
            Prompt(
                text="alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta theta",
                category=PromptCategory.REPETITIVE,
                description="Greek letters repetition",
                expected_pattern="Greek alphabet sequence repetition",
                metadata={"complexity": "low", "pattern": "greek_cycle"}
            ),
        ]
        
        # Add all prompts to the dataset
        self._prompts[PromptCategory.CODE] = code_prompts
        self._prompts[PromptCategory.JSON] = json_prompts
        self._prompts[PromptCategory.CONVERSATIONAL] = conversational_prompts
        self._prompts[PromptCategory.REPETITIVE] = repetitive_prompts
    
    def get_prompts(self, category: Optional[PromptCategory] = None) -> List[Prompt]:
        """
        Get prompts for a specific category or all categories.
        
        Args:
            category: Specific category to retrieve, or None for all
            
        Returns:
            List of Prompt objects
        """
        if category is None:
            # Return all prompts flattened
            all_prompts = []
            for cat_prompts in self._prompts.values():
                all_prompts.extend(cat_prompts)
            return all_prompts
        return self._prompts[category].copy()
    
    def get_all_prompts(self) -> List[Prompt]:
        """Get all prompts across all categories."""
        return self.get_prompts(category=None)
    
    def get_by_category_dict(self) -> Dict[str, List[Prompt]]:
        """Get prompts organized by category name."""
        return {cat.value: prompts for cat, prompts in self._prompts.items()}
    
    def add_prompt(self, prompt: Prompt):
        """Add a custom prompt to the dataset."""
        self._prompts[prompt.category].append(prompt)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset to dictionary."""
        return {
            "categories": {
                cat.value: [p.to_dict() for p in prompts]
                for cat, prompts in self._prompts.items()
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize dataset to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptDataset":
        """Create dataset from dictionary."""
        dataset = cls()
        # Clear default prompts
        dataset._prompts = {cat: [] for cat in PromptCategory}
        
        # Load from data
        for cat_name, prompts_data in data.get("categories", {}).items():
            category = PromptCategory(cat_name)
            for prompt_data in prompts_data:
                dataset._prompts[category].append(Prompt.from_dict(prompt_data))
        
        return dataset
    
    @classmethod
    def from_file(cls, path: str) -> "PromptDataset":
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_file(self, path: str):
        """Save dataset to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __len__(self) -> int:
        """Total number of prompts in dataset."""
        return sum(len(prompts) for prompts in self._prompts.values())
    
    def __repr__(self) -> str:
        return f"PromptDataset(total={len(self)}, by_category={ {cat.value: len(prompts) for cat, prompts in self._prompts.items()} })"


class TrainTestSplitter:
    """
    Implements train/test split methodology for n-gram cache building.
    
    Splits text or token sequences into training context (for building n-gram cache)
    and test continuation (for measuring acceptance rates).
    
    Default split ratio is 60/40, which provides sufficient training context
    while leaving enough test tokens for meaningful acceptance measurement.
    
    Usage:
        splitter = TrainTestSplitter(train_ratio=0.6)
        
        # Split text directly
        train_text, test_text = splitter.split_text(prompt_text)
        
        # Split token IDs
        train_ids, test_ids = splitter.split_tokens(token_ids)
        
        # Get full split object with all formats
        split = splitter.split(prompt_text, tokenizer)
    """
    
    def __init__(self, train_ratio: float = 0.6, min_train_length: int = 10, min_test_length: int = 5):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Ratio of tokens to use for training (0.0-1.0)
            min_train_length: Minimum training sequence length
            min_test_length: Minimum test sequence length
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        
        self.train_ratio = train_ratio
        self.min_train_length = min_train_length
        self.min_test_length = min_test_length
    
    def split_text(self, text: str) -> Tuple[str, str]:
        """
        Split text into train/test portions.
        
        Args:
            text: Input text to split
            
        Returns:
            Tuple of (train_text, test_text)
        """
        if len(text) < self.min_train_length + self.min_test_length:
            raise ValueError(
                f"Text too short for split (length={len(text)}, "
                f"min_required={self.min_train_length + self.min_test_length})"
            )
        
        split_idx = int(len(text) * self.train_ratio)
        
        # Ensure minimum lengths
        split_idx = max(split_idx, self.min_train_length)
        split_idx = min(split_idx, len(text) - self.min_test_length)
        
        return text[:split_idx], text[split_idx:]
    
    def split_tokens(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Split token IDs into train/test portions.
        
        Args:
            token_ids: Input token sequence to split
            
        Returns:
            Tuple of (train_token_ids, test_token_ids)
        """
        if len(token_ids) < self.min_train_length + self.min_test_length:
            raise ValueError(
                f"Token sequence too short for split (length={len(token_ids)}, "
                f"min_required={self.min_train_length + self.min_test_length})"
            )
        
        split_idx = int(len(token_ids) * self.train_ratio)
        
        # Ensure minimum lengths
        split_idx = max(split_idx, self.min_train_length)
        split_idx = min(split_idx, len(token_ids) - self.min_test_length)
        
        return token_ids[:split_idx], token_ids[split_idx:]
    
    def split(self, text: str, tokenizer=None) -> TrainTestSplit:
        """
        Create a full train/test split with both text and token representations.
        
        Args:
            text: Input text to split
            tokenizer: Optional tokenizer for converting text to tokens.
                      If None, uses simple character-level tokenization.
        
        Returns:
            TrainTestSplit object with all representations
        """
        # Split text
        train_text, test_text = self.split_text(text)
        
        # Tokenize
        if tokenizer is not None:
            # Use provided tokenizer
            if hasattr(tokenizer, 'encode'):
                train_token_ids = tokenizer.encode(train_text)
                test_token_ids = tokenizer.encode(test_text)
            elif hasattr(tokenizer, 'tokenize'):
                train_token_ids = [tokenizer.vocab.get(t, 0) for t in tokenizer.tokenize(train_text)]
                test_token_ids = [tokenizer.vocab.get(t, 0) for t in tokenizer.tokenize(test_text)]
            else:
                raise ValueError("Tokenizer must have 'encode' or 'tokenize' method")
        else:
            # Simple character-level tokenization (fallback)
            train_token_ids = [ord(c) % 256 for c in train_text]
            test_token_ids = [ord(c) % 256 for c in test_text]
        
        return TrainTestSplit(
            train_text=train_text,
            test_text=test_text,
            train_token_ids=train_token_ids,
            test_token_ids=test_token_ids,
            split_ratio=self.train_ratio
        )


class TokenizerAdapter:
    """
    Adapter for processing prompts with various tokenizers.
    
    Provides a unified interface for tokenizing prompts regardless of
    the underlying tokenizer implementation.
    
    Usage:
        # With HuggingFace tokenizer
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        adapter = TokenizerAdapter(hf_tokenizer)
        
        # Tokenize
        tokens = adapter.encode(prompt_text)
        text = adapter.decode(tokens)
        
        # Batch tokenize multiple prompts
        all_tokens = adapter.encode_batch(prompts)
    """
    
    def __init__(self, tokenizer):
        """
        Initialize with a tokenizer.
        
        Args:
            tokenizer: Tokenizer object with encode/decode methods
        """
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, 'tokenize'):
            return [self.tokenizer.vocab.get(t, 0) for t in self.tokenizer.tokenize(text)]
        else:
            # Fallback: character-level
            return [ord(c) % 256 for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids)
        elif hasattr(self.tokenizer, 'convert_ids_to_tokens'):
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            return ''.join(tokens)
        else:
            # Fallback: character-level
            return ''.join(chr(tid % 256) for tid in token_ids)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts to token IDs."""
        return [self.encode(text) for text in texts]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'vocab'):
            return len(self.tokenizer.vocab)
        else:
            return 256  # Fallback for character-level
    
    @classmethod
    def create_char_level(cls) -> "TokenizerAdapter":
        """Create a simple character-level tokenizer adapter."""
        class CharLevelTokenizer:
            def encode(self, text: str) -> List[int]:
                return [ord(c) % 256 for c in text]
            
            def decode(self, token_ids: List[int]) -> str:
                return ''.join(chr(tid % 256) for tid in token_ids)
        
        return cls(CharLevelTokenizer())


# Convenience functions for quick usage

def get_default_dataset() -> PromptDataset:
    """Get the default prompt dataset."""
    return PromptDataset()


def get_prompts_by_category(category: str) -> List[Prompt]:
    """Get prompts for a specific category by name."""
    dataset = PromptDataset()
    return dataset.get_prompts(PromptCategory(category))


def create_splitter(train_ratio: float = 0.6) -> TrainTestSplitter:
    """Create a train/test splitter with specified ratio."""
    return TrainTestSplitter(train_ratio=train_ratio)


def get_char_level_tokenizer() -> TokenizerAdapter:
    """Get a character-level tokenizer adapter."""
    return TokenizerAdapter.create_char_level()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 72)
    print("  Prompt Dataset Test")
    print("=" * 72)
    
    # Create dataset
    dataset = PromptDataset()
    print(f"\nDataset: {dataset}")
    
    # Test each category
    for category in PromptCategory:
        prompts = dataset.get_prompts(category)
        print(f"\n{category.value.upper()}: {len(prompts)} prompts")
        for i, prompt in enumerate(prompts[:2]):  # Show first 2
            print(f"  [{i+1}] {prompt.description}")
            print(f"      Text: {prompt.text[:60]}...")
    
    # Test train/test split
    print("\n" + "=" * 72)
    print("  Train/Test Split Test")
    print("=" * 72)
    
    splitter = TrainTestSplitter(train_ratio=0.6)
    test_text = "This is a test sentence for train/test split validation."
    
    train_text, test_text_result = splitter.split_text(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Train ({len(train_text)} chars): {train_text}")
    print(f"Test ({len(test_text_result)} chars): {test_text_result}")
    
    # Test tokenization
    print("\n" + "=" * 72)
    print("  Tokenizer Test")
    print("=" * 72)
    
    tokenizer = get_char_level_tokenizer()
    sample_text = "Hello, World!"
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nOriginal: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {sample_text == decoded}")
    
    # Test serialization
    print("\n" + "=" * 72)
    print("  Serialization Test")
    print("=" * 72)
    
    json_str = dataset.to_json()
    print(f"\nSerialized to JSON: {len(json_str)} bytes")
    
    # Deserialize
    dataset2 = PromptDataset.from_dict(json.loads(json_str))
    print(f"Deserialized: {dataset2}")
    print(f"Round-trip successful: {len(dataset.get_all_prompts()) == len(dataset2.get_all_prompts())}")
    
    print("\n" + "=" * 72)
    print("  All Tests Complete")
    print("=" * 72)
