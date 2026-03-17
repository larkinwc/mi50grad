"""
N-gram lookahead speculative decoding for Qwen 3.5 27B on MI50.

This module implements training-free speculative decoding using n-gram matches
from the existing prompt and generated text as draft tokens.

Algorithm:
1. Maintain a trie/hash of n-grams seen in the prompt + generated text so far
2. After generating token t, look up the (n-1)-gram ending with t in the trie
3. If found, the continuation tokens are draft candidates
4. Run the model forward pass on the draft sequence (multiple tokens at once via prefill-style)
5. Verify: compare model's output distribution at each position with draft tokens
6. Accept the longest matching prefix; reject and regenerate from first mismatch

For TP=4, the key win: the verification pass processes K draft tokens in one forward pass,
amortizing allreduce cost (10.1ms) across K tokens instead of 1.
"""

from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import numpy as np


class NgramCache:
    """Builds and queries n-gram trie from token sequences.
    
    The cache indexes all n-grams from the prompt and generated text so far,
    allowing fast lookup of continuation tokens given an (n-1)-gram context.
    
    Uses a nested dictionary structure for the trie:
    - Level 0: token_id -> {Level 1}
    - Level 1: token_id -> {Level 2}
    - ...
    - Level n-1: token_id -> list of continuation tokens
    
    For n=3, we store trigrams as:
    {t1: {t2: {t3: [continuation_tokens]}}}
    """
    
    def __init__(self, n: int = 3):
        """Initialize n-gram cache.
        
        Args:
            n: size of n-grams to track (default: 3)
        """
        self.n = n
        self.trie: Dict = {}
    
    def build_from_sequence(self, tokens: List[int]):
        """Build n-gram cache from a sequence of tokens.
        
        Args:
            tokens: list of token IDs to index
        """
        if len(tokens) < self.n:
            return
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            self._add_ngram(ngram)
    
    def _add_ngram(self, ngram: Tuple[int, ...]):
        """Add a single n-gram to the trie.
        
        Args:
            ngram: tuple of n token IDs
        """
        node = self.trie
        # Navigate/create path for first n-1 tokens
        for token in ngram[:-1]:
            if token not in node:
                node[token] = {}
            node = node[token]
        
        # Store continuation token at the leaf
        last_token = ngram[-1]
        if '_continuations' not in node:
            node['_continuations'] = []
        if last_token not in node['_continuations']:
            node['_continuations'].append(last_token)
    
    def query(self, context: List[int]) -> Optional[List[int]]:
        """Query for draft tokens given context.
        
        Looks up the (n-1)-gram ending with the last token in context,
        and returns continuation candidates.
        
        Args:
            context: list of tokens seen so far (prompt + generated)
        
        Returns:
            List of draft continuation tokens, or None if no match
        """
        if len(context) < self.n - 1:
            return None
        
        # Get the last (n-1) tokens as context
        context_tokens = tuple(context[-(self.n - 1):])
        
        # Navigate the trie
        node = self.trie
        for token in context_tokens:
            if token not in node:
                return None
            node = node[token]
        
        # Return continuations if available
        return node.get('_continuations', None)
    
    def update(self, new_tokens: List[int]):
        """Update cache with newly generated tokens.
        
        Args:
            new_tokens: list of new token IDs to add to cache
        """
        self.build_from_sequence(new_tokens)
    
    def clear(self):
        """Clear the n-gram cache."""
        self.trie = {}


def speculative_decode(generator,
                       input_ids: List[int],
                       params,
                       ngram_cache: NgramCache,
                       ngram_size: int = 3,
                       max_draft_len: int = 5,
                       token_id_key: str = 'token_id',
                       verbose: bool = False) -> Tuple[List[int], dict]:
    """Perform speculative decoding with n-gram lookahead.
    
    This function orchestrates the draft-verify loop:
    1. Build/update n-gram cache from current context
    2. Generate draft tokens using n-gram matches
    3. Verify drafts by running model forward pass
    4. Accept longest matching prefix, reject rest
    5. Repeat until stop condition
    
    Args:
        generator: TextGenerator instance with prefill() and _lm_head() methods
        input_ids: initial prompt token IDs
        params: SamplingParams for generation
        ngram_cache: NgramCache instance for draft token lookup
        ngram_size: size of n-grams to use (default: 3)
        max_draft_len: maximum number of draft tokens to generate per iteration
        verbose: if True, print debugging info
        token_id_key: attribute name for token ID in returned dict (for compatibility)
    
    Returns:
        Tuple of (generated_token_ids, stats_dict)
    """
    from src.inference.sampler import sample_token
    
    generated_ids = []
    position = len(input_ids)
    
    # Stats tracking
    total_drafts = 0
    total_accepted = 0
    total_iterations = 0
    
    # Generate up to max_tokens
    while len(generated_ids) < params.max_tokens:
        total_iterations += 1
        
        # Build context (prompt + generated so far)
        context = input_ids + generated_ids
        
        # Update n-gram cache with current context
        # Only need to update with the last token(s) generated
        if len(generated_ids) > 0:
            # Add recent context window to cache
            window_start = max(0, len(context) - ngram_size * 2)
            ngram_cache.build_from_sequence(context[window_start:])
        
        # Try to generate draft tokens from n-gram matches
        draft_tokens = []
        current_context = context.copy()
        
        # Limit draft length to not exceed max_tokens
        remaining = params.max_tokens - len(generated_ids)
        actual_max_draft = min(max_draft_len, remaining)
        
        for _ in range(actual_max_draft):
            candidates = ngram_cache.query(current_context)
            if candidates is None or len(candidates) == 0:
                break
            
            # Pick the first candidate as draft (could use sampling strategy)
            draft_token = candidates[0]
            draft_tokens.append(draft_token)
            current_context.append(draft_token)
        
        total_drafts += len(draft_tokens)
        
        if verbose and len(draft_tokens) > 0:
            print(f"  Draft {len(draft_tokens)} tokens: {draft_tokens}")
        
        # Verify drafts by running model forward pass
        # For now, use simple verification: check if model would generate same tokens
        accepted_tokens = []
        hidden = generator.prefill(context)
        
        verify_pos = position
        all_match = True
        
        for i, draft_token in enumerate(draft_tokens):
            # Get model's logits at current position
            logits = generator._lm_head(hidden)
            
            # Check if draft token matches model's greedy choice
            model_choice = int(np.argmax(logits))
            
            if model_choice == draft_token:
                accepted_tokens.append(draft_token)
                # Generate next hidden state
                emb = generator._embed_token(draft_token)
                hidden = generator.engine.decode_step(emb, verify_pos)
                verify_pos += 1
            else:
                # Mismatch - reject from this point
                if verbose:
                    print(f"  Rejected at position {i}: draft={draft_token}, model={model_choice}")
                all_match = False
                break
        
        total_accepted += len(accepted_tokens)
        
        if len(accepted_tokens) > 0:
            # We have some accepted tokens
            if verbose:
                print(f"  Accepted {len(accepted_tokens)} of {len(draft_tokens)} drafts")
            
            generated_ids.extend(accepted_tokens)
            position += len(accepted_tokens)
            
            # If all drafts were accepted, we need to generate the NEXT token
            # This token should match what standard greedy would produce at this position
            if all_match and len(draft_tokens) > 0:
                # All drafts accepted - generate next token to continue
                logits = generator._lm_head(hidden)
                if params.temperature == 0:
                    next_token = int(np.argmax(logits))
                else:
                    next_token = sample_token(logits, params, past_tokens=input_ids + generated_ids)
                
                # Don't add this token yet - it will be verified in next iteration
                # Just update hidden state
                emb = generator._embed_token(next_token)
                hidden = generator.engine.decode_step(emb, position)
                position += 1
            else:
                # Some drafts rejected - the first rejected position needs a new token
                # Get model's choice at the position after accepted prefix
                logits = generator._lm_head(hidden)
                if params.temperature == 0:
                    token_id = int(np.argmax(logits))
                else:
                    token_id = sample_token(logits, params, past_tokens=input_ids + generated_ids)
                
                if params.stop_token_ids and token_id in params.stop_token_ids:
                    break
                if token_id == getattr(generator.tokenizer, 'eos_token_id', None):
                    break
                
                generated_ids.append(token_id)
                emb = generator._embed_token(token_id)
                hidden = generator.engine.decode_step(emb, position)
                position += 1
        else:
            # All drafts rejected - use standard greedy decode (identical to baseline)
            if verbose:
                print(f"  All drafts rejected, using standard greedy")
            
            logits = generator._lm_head(hidden)
            if params.temperature == 0:
                token_id = int(np.argmax(logits))
            else:
                token_id = sample_token(logits, params, past_tokens=input_ids + generated_ids)
            
            if params.stop_token_ids and token_id in params.stop_token_ids:
                break
            if token_id == getattr(generator.tokenizer, 'eos_token_id', None):
                break
            
            generated_ids.append(token_id)
            emb = generator._embed_token(token_id)
            hidden = generator.engine.decode_step(emb, position)
            position += 1
        
        # Check stop conditions
        if len(generated_ids) > 0 and params.stop_token_ids:
            last_token = generated_ids[-1]
            if last_token in params.stop_token_ids:
                break
        
        # Check max_tokens
        if len(generated_ids) >= params.max_tokens:
            break
    
    stats = {
        'total_drafts': total_drafts,
        'total_accepted': total_accepted,
        'total_iterations': total_iterations,
        'acceptance_rate': total_accepted / total_drafts if total_drafts > 0 else 0.0,
    }
    
    return generated_ids, stats


class SpeculativeGenerator:
    """Wrapper around TextGenerator that adds speculative decoding support."""
    
    def __init__(self, engine, embed_weight: np.ndarray, lm_head_weight: np.ndarray,
                 tokenizer=None, ngram_size: int = 3, max_draft_len: int = 5):
        """Initialize speculative generator.
        
        Args:
            engine: InferenceEngine instance
            embed_weight: embedding weight array
            lm_head_weight: LM head weight array
            tokenizer: tokenizer instance
            ngram_size: size of n-grams for draft generation
            max_draft_len: maximum draft tokens per iteration
        """
        from src.inference.generate import TextGenerator
        
        self.generator = TextGenerator(engine, embed_weight, lm_head_weight, tokenizer)
        self.ngram_cache = NgramCache(n=ngram_size)
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len
    
    def generate(self, prompt: str, params=None, verbose: bool = False):
        """Generate text using speculative decoding.
        
        Args:
            prompt: input text
            params: sampling parameters
            verbose: if True, print debugging info
        
        Returns:
            Tuple of (generated_text, stats_dict)
        """
        if params is None:
            params = self.generator.engine.params if hasattr(self.generator.engine, 'params') else None
            if params is None:
                from src.inference.sampler import SamplingParams
                params = SamplingParams()
        
        if self.generator.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")
        
        input_ids = self.generator.tokenizer.encode(prompt)
        
        # Build initial n-gram cache from prompt
        self.ngram_cache.build_from_sequence(input_ids)
        
        # Run speculative decode
        generated_ids, stats = speculative_decode(
            self.generator,
            input_ids,
            params,
            self.ngram_cache,
            ngram_size=self.ngram_size,
            max_draft_len=self.max_draft_len,
            verbose=verbose
        )
        
        return self.generator.tokenizer.decode(generated_ids), stats
    
    def benchmark(self, prompt: str, max_tokens: int = 100, verbose: bool = False):
        """Benchmark speculative decoding vs standard decode.
        
        Args:
            prompt: input text
            max_tokens: maximum tokens to generate
            verbose: if True, print debugging info
        
        Returns:
            Dict with timing and acceptance stats
        """
        import time
        from src.inference.sampler import SamplingParams
        
        params = SamplingParams(temperature=0, max_tokens=max_tokens)
        
        # Standard decode
        t0 = time.perf_counter()
        self.generator.params = params  # Set params for generator
        standard_text = self.generator.generate(prompt, params)
        standard_time = time.perf_counter() - t0
        
        # Speculative decode
        t0 = time.perf_counter()
        spec_text, spec_stats = self.generate(prompt, params, verbose=verbose)
        spec_time = time.perf_counter() - t0
        
        return {
            'standard_time_ms': standard_time * 1000,
            'speculative_time_ms': spec_time * 1000,
            'speedup': standard_time / spec_time if spec_time > 0 else 1.0,
            'acceptance_rate': spec_stats['acceptance_rate'],
            'total_drafts': spec_stats['total_drafts'],
            'total_accepted': spec_stats['total_accepted'],
            'output_match': standard_text == spec_text,
        }
    
    def cleanup(self):
        """Free resources."""
        self.generator.cleanup()
