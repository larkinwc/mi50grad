# Speculative Decode Implementation Notes

## NgramCache Behavior

The `NgramCache` class in `src/inference/speculative.py` uses a trie structure with `_continuations` lists at leaf nodes. Key behaviors to be aware of:

### Multiple Continuations Per Context

When building the cache, multiple continuation tokens can be stored for the same (n-1)-gram context:

```python
# Line 47 in speculative.py
if last_token not in node['_continuations']:
    node['_continuations'].append(last_token)
```

This allows the cache to return multiple possible draft candidates for a given context. When measuring acceptance rates:

- Counting **matches** (whether the context has continuations) can result in >100% "acceptance" because every n-gram in the training sequence produces a match
- **Proper acceptance measurement** requires running actual speculative decode and counting accepted draft tokens vs total draft tokens generated

### Training-on-Test Pitfall

When validating n-gram acceptance rates, do NOT build the cache from the same sequence you're measuring acceptance on. This inflates acceptance to meaningless levels.

**Incorrect (training-on-test):**
```python
input_ids = tokenize_simple(prompt)
ngram_cache.build_from_sequence(input_ids)  # Build from same sequence
# Then query against the same sequence
for i in range(len(input_ids)):
    context = input_ids[max(0, i-ngram_size+1):i]
    if ngram_cache.query(context) is not None:
        matches += 1  # Will be nearly 100% because all n-grams were indexed
```

**Correct approach:**
```python
# Build cache from prompt prefix
prefix_ids = tokenize_simple(prompt_prefix)
ngram_cache.build_from_sequence(prefix_ids)

# Measure acceptance on continuation text
continuation_ids = tokenize_real(prompt_continuation)
# Run actual speculative decode and count accepted/total draft tokens
```

## EAGLE Draft Head

The `EagleDraftHead` class provides training-free speculative decoding by reusing the model's LM head and embedding weights. Validation should:

1. Run actual speculative decode with draft head
2. Measure `total_accepted / total_drafts` 
3. Compare timing against standard greedy decode for speedup

Do NOT just check that the class exists - the validation contract requires actual acceptance rate measurements.
