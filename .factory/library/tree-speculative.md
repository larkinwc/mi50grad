# Tree-Based Speculative Decoding (Sequoia-style)

Design notes for tree-structured speculative verification.

**What belongs here:** Tree attention masks, DP tree optimization, tree verification pipeline.

---

## Core Concept

Instead of verifying K draft tokens sequentially (K decode steps, each with 64 allreduces), verify them all in a single forward pass using tree-structured attention. Cost: 1 decode step (64 allreduces). Benefit: accept multiple tokens from the tree.

## Tree Attention Mask

A tree of draft tokens has parent-child relationships. Each node attends to:
- All its ancestors (path from root to node)
- Itself
- KV cache (all previously generated tokens)

Example tree (depth 3, branching 2):
```
root(0)
├── child(1)
│   ├── grandchild(3)
│   └── grandchild(4)
└── child(2)
    ├── grandchild(5)
    └── grandchild(6)
```

Attention mask (tree tokens only, excluding KV cache which all attend to):
```
     0  1  2  3  4  5  6
0  [ 1  0  0  0  0  0  0 ]  # root sees only itself
1  [ 1  1  0  0  0  0  0 ]  # child1 sees root + itself
2  [ 1  0  1  0  0  0  0 ]  # child2 sees root + itself
3  [ 1  1  0  1  0  0  0 ]  # gc3 sees root + child1 + itself
4  [ 1  1  0  0  1  0  0 ]  # gc4 sees root + child1 + itself
5  [ 1  0  1  0  0  1  0 ]  # gc5 sees root + child2 + itself
6  [ 1  0  1  0  0  0  1 ]  # gc6 sees root + child2 + itself
```

## DP Tree Optimizer (Sequoia Algorithm)

Find tree topology T that maximizes E[accepted_tokens(T)].

**Recurrence:**
- For a node at depth d with children c1...cB:
  - E[tokens(node)] = p(d) * (1 + sum(E[tokens(ci)]))
  - Where p(d) is acceptance probability at depth d

**Position-dependent acceptance:**
Our measured rates: ~54% overall. Typically:
- Depth 1: ~60% (first continuation is easiest to predict)
- Depth 2: ~50%
- Depth 3: ~40%
- Depth 4: ~30%

**Budget constraint:** Total tree size <= B (limited by GEMM M dimension and memory).

**Hardware cost model:**
- Verification cost = 1 forward pass = 18.5ms (constant for tree sizes up to ~16)
- For larger trees, GEMM at M>1 is slower than GEMV at M=1, so there's a cost curve
- Our measured batch=2-3 showed no throughput degradation for M=2-3 (within noise)
- For M=4-16, need to measure GEMM latency scaling

## Existing Infrastructure to Leverage

- `src/inference/speculative.py`: NGramSpeculativeDecoder (n-gram draft generation)
- `tests/test_ngram_local.py`: N-gram acceptance rate testing
- `src/inference/tp_engine.py`: batch_decode_step (tested at M=2-3)
- `src/kernels/gemm_int4_prefill_v2.hip`: INT4 GEMM kernel for M>1
- `data/test_prompts.json`: Test prompt corpus (4 domains, 20 prompts)

## Verification Pipeline

1. Generate draft tokens using n-gram predictor in tree structure
2. Build tree embeddings: [tree_size, hidden_size] tensor
3. Build tree attention mask: [tree_size, tree_size + kv_len] boolean
4. Forward pass: batch_decode_step_tree(embeddings, mask) -> [tree_size, vocab_size] logits
5. Verify: walk tree from root, compare model logits vs draft token at each node
6. Accept: longest correct prefix path in tree
7. Update KV cache with accepted tokens, discard rejected branches

## Key Differences from Our Previous Speculative Attempts

Previous flat speculative decode (RESEARCH.md Section 6) showed 0% throughput gain because:
- Each draft token was verified in a SEPARATE decode step (separate allreduce set)
- Allreduce cost scaled linearly with number of draft tokens

Tree approach fixes this:
- ALL draft tokens verified in ONE decode step (one allreduce set)
- Amortizes allreduce across multiple accepted tokens
- E.g., accepting 3 tokens per verification step = 3x effective throughput for same allreduce cost
