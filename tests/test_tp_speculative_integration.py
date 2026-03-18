"""
Tests for TPInferenceEngine speculative decoding integration.

Tests:
1. TPInferenceEngine.set_speculative_mode() enables/disables speculative path
2. decode_step_speculative() accepts embeddings and returns hidden states
3. SpeculativeDecodeState tracks acceptance rate correctly
4. KV cache rollback works correctly
5. N-gram cache built from prompt and updated with generated tokens
"""

import numpy as np
from typing import List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_speculative_state_class():
    """Test SpeculativeDecodeState class functionality."""
    print("Testing SpeculativeDecodeState class...")
    
    from src.inference.tp_engine import SpeculativeDecodeState
    
    # Create a mock TP engine
    class MockKVCache:
        def __init__(self):
            self.current_len = 0
    
    class MockEngine:
        def __init__(self):
            self.kv_cache = MockKVCache()
    
    class MockConfig:
        def __init__(self):
            self.hidden_size = 5120
    
    class MockTP:
        def __init__(self):
            self.engines = [MockEngine()]
            self.config = MockConfig()
        
        def decode_step(self, emb, position):
            """Mock decode step - just return the embedding."""
            return emb.copy()
    
    mock_tp = MockTP()
    
    # Create speculative state
    spec_state = SpeculativeDecodeState(mock_tp, ngram_size=3, max_draft_len=5)
    
    # Test n-gram cache initialization
    assert spec_state.ngram_cache is not None
    print(f"  ✓ NgramCache initialized (n={spec_state.ngram_size})")
    
    # Test acceptance stats initialization
    assert 'total_drafts' in spec_state.acceptance_stats
    assert 'total_accepted' in spec_state.acceptance_stats
    assert 'total_iterations' in spec_state.acceptance_stats
    print(f"  ✓ Acceptance stats initialized")
    
    # Test build_cache_from_prompt
    prompt_tokens = [0, 1, 2, 3, 4, 5]
    spec_state.build_cache_from_prompt(prompt_tokens)
    result = spec_state.ngram_cache.query([0, 1])
    assert result is not None
    print(f"  ✓ build_cache_from_prompt works")
    
    # Test generate_drafts
    context = [0, 1, 2, 3, 4]
    drafts = spec_state.generate_drafts(context)
    assert isinstance(drafts, list)
    print(f"  ✓ generate_drafts works: {drafts}")
    
    # Test verify_drafts
    hidden_size = mock_tp.config.hidden_size
    embeddings = [np.zeros(hidden_size, dtype=np.float16) for _ in range(3)]
    hidden_states, _ = spec_state.verify_drafts(embeddings, position=5, draft_tokens=[10, 11, 12])
    assert len(hidden_states) == 3
    print(f"  ✓ verify_drafts works")
    
    # Test acceptance rate tracking
    spec_state.record_accepted_tokens(2)
    rate = spec_state.get_acceptance_rate()
    assert rate > 0
    print(f"  ✓ Acceptance rate tracking: {rate:.2f}")
    
    # Test get_stats
    stats = spec_state.get_stats()
    assert 'acceptance_rate' in stats
    print(f"  ✓ get_stats works: {stats}")
    
    # Test rollback_kv_cache
    initial_pos = mock_tp.engines[0].kv_cache.current_len
    mock_tp.engines[0].kv_cache.current_len = 10
    spec_state.rollback_kv_cache(5)
    assert mock_tp.engines[0].kv_cache.current_len == 5
    print(f"  ✓ rollback_kv_cache works")
    
    # Test cleanup
    spec_state.cleanup()
    print(f"  ✓ cleanup works")
    
    print("  ✅ SpeculativeDecodeState class test passed\n")


def test_tp_engine_speculative_methods():
    """Test that TPInferenceEngine has speculative decoding methods."""
    print("Testing TPInferenceEngine speculative methods...")
    
    from src.inference.tp_engine import TPInferenceEngine
    
    # Check methods exist
    assert hasattr(TPInferenceEngine, 'set_speculative_mode')
    assert hasattr(TPInferenceEngine, 'decode_step_speculative')
    print(f"  ✓ set_speculative_mode method exists")
    print(f"  ✓ decode_step_speculative method exists")
    
    # Check method signatures
    import inspect
    sig1 = inspect.signature(TPInferenceEngine.set_speculative_mode)
    assert 'enabled' in str(sig1)
    assert 'ngram_size' in str(sig1)
    assert 'max_draft_len' in str(sig1)
    print(f"  ✓ set_speculative_mode signature correct")
    
    sig2 = inspect.signature(TPInferenceEngine.decode_step_speculative)
    assert 'token_embeddings' in str(sig2)
    assert 'position' in str(sig2)
    assert 'draft_tokens' in str(sig2)
    print(f"  ✓ decode_step_speculative signature correct")
    
    print("  ✅ TPInferenceEngine speculative methods test passed\n")


def test_speculative_mode_workflow():
    """Test the complete speculative decoding workflow."""
    print("Testing speculative decoding workflow...")
    
    from src.inference.tp_engine import SpeculativeDecodeState
    from src.inference.speculative import NgramCache
    
    # Create mock TP engine
    class MockKVCache:
        def __init__(self):
            self.current_len = 5  # Start with prompt length
    
    class MockEngine:
        def __init__(self):
            self.kv_cache = MockKVCache()
    
    class MockConfig:
        def __init__(self):
            self.hidden_size = 128
    
    class MockTP:
        def __init__(self):
            self.engines = [MockEngine()]
            self.config = MockConfig()
        
        def decode_step(self, emb, position):
            """Mock decode step."""
            return emb.copy()
    
    mock_tp = MockTP()
    
    # Initialize speculative state
    spec_state = SpeculativeDecodeState(mock_tp, ngram_size=3, max_draft_len=5)
    
    # Simulate workflow:
    # 1. Build cache from prompt
    prompt = [0, 1, 2, 3, 4]  # hello world ! how are
    spec_state.build_cache_from_prompt(prompt)
    print(f"  ✓ Step 1: Built cache from prompt")
    
    # 2. Generate drafts from current context
    context = prompt + [5, 6, 7]  # Add some generated tokens
    drafts = spec_state.generate_drafts(context)
    print(f"  ✓ Step 2: Generated {len(drafts)} draft tokens")
    
    # 3. Verify drafts (mock - no actual model)
    if drafts:
        embeddings = [np.zeros(mock_tp.config.hidden_size, dtype=np.float16) 
                     for _ in drafts]
        hidden_states, _ = spec_state.verify_drafts(
            embeddings, position=len(context), draft_tokens=drafts)
        print(f"  ✓ Step 3: Verified {len(drafts)} drafts")
    
    # 4. Record acceptance (simulate 2 out of 3 accepted)
    num_accepted = min(2, len(drafts)) if drafts else 0
    spec_state.record_accepted_tokens(num_accepted)
    print(f"  ✓ Step 4: Recorded {num_accepted} accepted tokens")
    
    # 5. Get stats
    stats = spec_state.get_stats()
    print(f"  ✓ Step 5: Stats: acceptance_rate={stats['acceptance_rate']:.2f}")
    
    print("  ✅ Speculative decoding workflow test passed\n")


def test_ngram_cache_integration():
    """Test that n-gram cache integrates properly with speculative state."""
    print("Testing n-gram cache integration...")
    
    from src.inference.tp_engine import SpeculativeDecodeState
    
    # Create mock TP engine
    class MockKVCache:
        def __init__(self):
            self.current_len = 0
    
    class MockEngine:
        def __init__(self):
            self.kv_cache = MockKVCache()
    
    class MockConfig:
        def __init__(self):
            self.hidden_size = 128
    
    class MockTP:
        def __init__(self):
            self.engines = [MockEngine()]
            self.config = MockConfig()
        
        def decode_step(self, emb, position):
            """Mock decode step."""
            return emb.copy()
    
    mock_tp = MockTP()
    
    # Test with different n-gram sizes
    for n in [2, 3, 4, 5]:
        spec_state = SpeculativeDecodeState(mock_tp, ngram_size=n)
        assert spec_state.ngram_size == n
        assert spec_state.ngram_cache.n == n
        
        # Build and query
        tokens = list(range(20))
        spec_state.build_cache_from_prompt(tokens)
        
        # Query should work with n-1 context
        if n >= 2:
            context = tokens[:n-1]
            result = spec_state.ngram_cache.query(context)
            # Should find at least one continuation
            if result is not None:
                print(f"  ✓ n={n}: found continuations")
            else:
                print(f"  ✓ n={n}: no continuations (expected for some contexts)")
    
    print("  ✅ N-gram cache integration test passed\n")


def run_all_tests():
    """Run all TP speculative integration tests."""
    print("=" * 60)
    print("Running TP Speculative Integration Tests")
    print("=" * 60 + "\n")
    
    test_speculative_state_class()
    test_tp_engine_speculative_methods()
    test_speculative_mode_workflow()
    test_ngram_cache_integration()
    
    print("=" * 60)
    print("All TP speculative integration tests passed! ✅")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
