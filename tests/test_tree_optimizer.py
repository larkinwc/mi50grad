#!/usr/bin/env python3
"""
Tree Optimizer Tests.

This test suite validates the SequoiaTreeOptimizer implementation for
finding optimal tree topologies in speculative decoding.

Tests cover:
1. Basic optimizer functionality
2. DP algorithm correctness
3. Tree topology validation
4. Expected token calculations
5. Position-dependent acceptance rates
6. Performance benchmarks
7. Comparison against flat speculation
"""

import sys
import time
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from inference.tree_optimizer import (
    SequoiaTreeOptimizer,
    TreeTopology,
    create_position_dependent_optimizer,
)


def test_optimizer_initialization():
    """Test basic optimizer initialization."""
    print("=" * 80)
    print("TEST 1: Optimizer Initialization")
    print("=" * 80)
    
    # Test default initialization
    optimizer = SequoiaTreeOptimizer()
    assert optimizer.base_acceptance_rate == 0.54
    assert optimizer.max_depth == 4
    assert optimizer.max_branching == 2
    assert optimizer.size_budget == 16
    assert optimizer.acceptance_decay == 0.0
    print("  Default initialization: PASS")
    
    # Test custom initialization
    optimizer_custom = SequoiaTreeOptimizer(
        base_acceptance_rate=0.60,
        max_depth=5,
        max_branching=3,
        size_budget=32,
        acceptance_decay=0.1,
    )
    assert optimizer_custom.base_acceptance_rate == 0.60
    assert optimizer_custom.max_depth == 5
    assert optimizer_custom.max_branching == 3
    assert optimizer_custom.size_budget == 32
    assert optimizer_custom.acceptance_decay == 0.1
    print("  Custom initialization: PASS")
    
    # Test hardware cost calculation
    expected_cost = 64 * 46.0 / 1000.0  # 2.944ms
    assert abs(optimizer.verification_cost_ms - expected_cost) < 1e-6
    print(f"  Verification cost: {optimizer.verification_cost_ms:.3f}ms (expected {expected_cost:.3f}ms): PASS")
    
    # Test acceptance probability computation
    probs = optimizer_custom._acceptance_probs
    assert len(probs) == 6  # depth 0-5
    # Check decay: p(d) = 0.6 * (0.9)^d
    expected_p0 = 0.6
    expected_p1 = 0.6 * 0.9
    expected_p2 = 0.6 * 0.9 * 0.9
    assert abs(probs[0] - expected_p0) < 1e-6
    assert abs(probs[1] - expected_p1) < 1e-6
    assert abs(probs[2] - expected_p2) < 1e-6
    print("  Acceptance probabilities (with decay): PASS")
    
    print("TEST 1: ALL PASSED\n")
    return True


def test_acceptance_probabilities():
    """Test acceptance probability computation."""
    print("=" * 80)
    print("TEST 2: Acceptance Probabilities")
    print("=" * 80)
    
    # Test uniform acceptance
    optimizer_uniform = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        acceptance_decay=0.0,
        max_depth=5,
    )
    for d in range(6):
        prob = optimizer_uniform.get_acceptance_prob(d)
        assert abs(prob - 0.54) < 1e-6, f"Depth {d}: expected 0.54, got {prob}"
    print("  Uniform acceptance (all depths): PASS")
    
    # Test exponential decay
    optimizer_decay = SequoiaTreeOptimizer(
        base_acceptance_rate=0.60,
        acceptance_decay=0.1,
        max_depth=5,
    )
    # p(0) = 0.6
    assert abs(optimizer_decay.get_acceptance_prob(0) - 0.6) < 1e-6
    # p(1) = 0.6 * 0.9 = 0.54
    assert abs(optimizer_decay.get_acceptance_prob(1) - 0.54) < 1e-6
    # p(2) = 0.6 * 0.9^2 = 0.486
    assert abs(optimizer_decay.get_acceptance_prob(2) - 0.486) < 1e-6
    # p(5) = 0.6 * 0.9^5 = 0.354...
    expected_p5 = 0.6 * (0.9 ** 5)
    assert abs(optimizer_decay.get_acceptance_prob(5) - expected_p5) < 1e-6
    print("  Exponential decay: PASS")
    
    # Test out-of-bounds depth
    prob_out = optimizer_decay.get_acceptance_prob(10)
    assert prob_out == 0.0
    print("  Out-of-bounds depth (returns 0): PASS")
    
    print("TEST 2: ALL PASSED\n")
    return True


def test_subtree_expectation():
    """Test subtree expectation calculation."""
    print("=" * 80)
    print("TEST 3: Subtree Expectation Calculation")
    print("=" * 80)
    
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.5,  # Easy to calculate
        acceptance_decay=0.0,
    )
    
    # Test leaf node (no children)
    leaf_exp = optimizer._compute_subtree_expectation(0, [])
    # E = p(0) * (1 + 0) = 0.5
    assert abs(leaf_exp - 0.5) < 1e-6
    print("  Leaf node (depth 0): E = 0.5: PASS")
    
    # Test node with one child
    # E = p(0) * (1 + E[child])
    # If child is leaf at depth 1: E[child] = 0.5
    # E = 0.5 * (1 + 0.5) = 0.75
    one_child_exp = optimizer._compute_subtree_expectation(
        0,
        [0.5]
    )
    assert abs(one_child_exp - 0.75) < 1e-6
    print("  One child: E = 0.75: PASS")
    
    # Test node with two children
    # E = 0.5 * (1 + 0.5 + 0.5) = 1.0
    two_children_exp = optimizer._compute_subtree_expectation(
        0,
        [0.5, 0.5]
    )
    assert abs(two_children_exp - 1.0) < 1e-6
    print("  Two children: E = 1.0: PASS")
    
    print("TEST 3: ALL PASSED\n")
    return True


def test_dp_solver():
    """Test DP solver correctness."""
    print("=" * 80)
    print("TEST 4: DP Solver")
    print("=" * 80)
    
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.5,
        max_depth=3,
        max_branching=2,
        size_budget=4,
    )
    
    # Test with budget 1 (just root)
    exp, children = optimizer._solve_dp(1, 0)
    assert exp == 0.5  # Leaf at depth 0
    assert children == []
    print("  Budget=1 (leaf): E=0.5: PASS")
    
    # Test with budget 2 (root + 1 child)
    exp, children = optimizer._solve_dp(2, 0)
    # Optimal: root with 1 child (leaf at depth 1)
    # E = 0.5 * (1 + 0.5) = 0.75
    assert abs(exp - 0.75) < 1e-6
    assert len(children) == 1
    print(f"  Budget=2 (root+child): E={exp:.3f}: PASS")
    
    # Test caching
    optimizer._dp_cache.clear()
    exp1, _ = optimizer._solve_dp(3, 0)
    cache_size_after_first = len(optimizer._dp_cache)
    
    exp2, _ = optimizer._solve_dp(3, 0)
    cache_size_after_second = len(optimizer._dp_cache)
    
    assert exp1 == exp2
    assert cache_size_after_second == cache_size_after_first
    print("  DP caching works: PASS")
    
    print("TEST 4: ALL PASSED\n")
    return True


def test_tree_topology():
    """Test TreeTopology data structure."""
    print("=" * 80)
    print("TEST 5: TreeTopology Data Structure")
    print("=" * 80)
    
    # Test basic topology
    edges = [(0, 1), (0, 2), (1, 3)]
    topology = TreeTopology(edges=edges, num_nodes=4, expected_tokens=2.5)
    
    assert topology.edges == edges
    assert topology.num_nodes == 4
    assert abs(topology.expected_tokens - 2.5) < 1e-6
    print("  Basic topology creation: PASS")
    
    # Test serialization
    topo_dict = topology.to_dict()
    assert topo_dict['edges'] == edges
    assert topo_dict['num_nodes'] == 4
    assert abs(topo_dict['expected_tokens'] - 2.5) < 1e-6
    print("  Serialization to dict: PASS")
    
    print("TEST 5: ALL PASSED\n")
    return True


def test_find_optimal_tree():
    """Test finding optimal tree topology."""
    print("=" * 80)
    print("TEST 6: Find Optimal Tree")
    print("=" * 80)
    
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        max_depth=4,
        max_branching=2,
        size_budget=16,
    )
    
    tree = optimizer.find_optimal_tree()
    
    # Verify tree structure
    assert tree.num_nodes >= 1
    assert tree.num_nodes <= optimizer.size_budget
    print(f"  Tree size: {tree.num_nodes} (budget: {optimizer.size_budget}): PASS")
    
    # Verify edges are valid
    if tree.edges:
        # All node indices should be within bounds
        max_node = max(max(e) for e in tree.edges)
        assert max_node < tree.num_nodes
        print("  Edge indices valid: PASS")
        
        # Check for cycles (simple check: no node appears as child twice)
        children = [e[1] for e in tree.edges]
        assert len(children) == len(set(children)), "Tree has cycles or duplicate children"
        print("  No cycles/duplicates: PASS")
    
    # Verify expected tokens
    assert tree.expected_tokens > 0
    print(f"  Expected tokens: {tree.expected_tokens:.3f}: PASS")
    
    print("TEST 6: ALL PASSED\n")
    return True


def test_expected_tokens_target():
    """Test that optimizer achieves E[accepted] >= 2.0 at 54% acceptance."""
    print("=" * 80)
    print("TEST 7: Expected Tokens Target (>= 2.0 at 54%)")
    print("=" * 80)
    
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        max_depth=4,
        max_branching=2,
        size_budget=16,
    )
    
    tree = optimizer.find_optimal_tree()
    
    print(f"  Optimal tree E[accepted]: {tree.expected_tokens:.4f}")
    print(f"  Target: >= 2.0")
    
    assert tree.expected_tokens >= 2.0, \
        f"Expected tokens {tree.expected_tokens:.4f} < 2.0"
    print("  Target ACHIEVED: PASS")
    
    # Additional check: verify it's actually better than just root
    root_only_exp = 0.54  # Just the root
    assert tree.expected_tokens > root_only_exp
    print(f"  Better than root-only ({root_only_exp}): PASS")
    
    print("TEST 7: ALL PASSED\n")
    return True


def test_position_dependent_optimizer():
    """Test position-dependent acceptance rates."""
    print("=" * 80)
    print("TEST 8: Position-Dependent Acceptance Rates")
    print("=" * 80)
    
    # Test code domain (59% base)
    optimizer_code = create_position_dependent_optimizer(code_domain=True)
    assert optimizer_code.base_acceptance_rate == 0.59
    assert optimizer_code.acceptance_decay > 0
    tree_code = optimizer_code.find_optimal_tree()
    print(f"  Code domain (59% base, decay): E[accepted] = {tree_code.expected_tokens:.3f}: PASS")
    
    # Test repetitive domain (87% base)
    optimizer_rep = create_position_dependent_optimizer(repetitive_domain=True)
    assert optimizer_rep.base_acceptance_rate == 0.87
    assert optimizer_rep.acceptance_decay > 0
    tree_rep = optimizer_rep.find_optimal_tree()
    print(f"  Repetitive domain (87% base, decay): E[accepted] = {tree_rep.expected_tokens:.3f}: PASS")
    
    # Test default (54% uniform)
    optimizer_default = create_position_dependent_optimizer()
    assert optimizer_default.base_acceptance_rate == 0.54
    assert optimizer_default.acceptance_decay == 0.0
    tree_default = optimizer_default.find_optimal_tree()
    print(f"  Default (54% uniform): E[accepted] = {tree_default.expected_tokens:.3f}: PASS")
    
    # Verify ordering: repetitive > code > default
    assert tree_rep.expected_tokens > tree_code.expected_tokens
    assert tree_code.expected_tokens > tree_default.expected_tokens
    print("  Ordering (rep > code > default): PASS")
    
    print("TEST 8: ALL PASSED\n")
    return True


def test_compare_with_chain():
    """Test comparison against flat (linear chain) speculation."""
    print("=" * 80)
    print("TEST 9: Comparison with Flat Chain Speculation")
    print("=" * 80)
    
    # Test 1: Uniform acceptance
    print("\n  Test 1: Uniform acceptance (54%)")
    optimizer_uniform = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        max_depth=4,
        max_branching=2,
        size_budget=16,
    )
    
    comparison_uniform = optimizer_uniform.compare_with_chain()
    tree_exp_uniform = comparison_uniform['optimal_tree']['expected_tokens']
    chain_exp_uniform = comparison_uniform['chain']['expected_tokens']
    improvement_uniform = comparison_uniform['improvement']['relative']
    
    print(f"    Tree E[accepted]: {tree_exp_uniform:.3f}")
    print(f"    Chain E[accepted]: {chain_exp_uniform:.3f}")
    print(f"    Tree improvement: {improvement_uniform:.1f}%")
    
    # With uniform acceptance and max_depth constraint, tree should be better
    # because it can accept multiple branches in parallel
    assert tree_exp_uniform >= chain_exp_uniform, \
        f"Tree ({tree_exp_uniform:.3f}) should be >= chain ({chain_exp_uniform:.3f})"
    print("    Tree >= chain: PASS")
    
    # Test 2: Position-dependent decay (tree advantage should be larger)
    print("\n  Test 2: Position-dependent decay (60% base, 15% decay)")
    optimizer_decay = SequoiaTreeOptimizer(
        base_acceptance_rate=0.60,
        max_depth=4,
        max_branching=2,
        size_budget=16,
        acceptance_decay=0.15,
    )
    
    comparison_decay = optimizer_decay.compare_with_chain()
    tree_exp_decay = comparison_decay['optimal_tree']['expected_tokens']
    chain_exp_decay = comparison_decay['chain']['expected_tokens']
    improvement_decay = comparison_decay['improvement']['relative']
    
    print(f"    Tree E[accepted]: {tree_exp_decay:.3f}")
    print(f"    Chain E[accepted]: {chain_exp_decay:.3f}")
    print(f"    Improvement: {improvement_decay:.1f}%")
    
    # With decay, tree should still be better
    assert tree_exp_decay >= chain_exp_decay - 1e-6, \
        f"Tree ({tree_exp_decay:.3f}) should be >= chain ({chain_exp_decay:.3f})"
    print("    Tree >= chain: PASS")
    
    print("  Overall comparison completed: PASS")
    
    print("TEST 9: ALL PASSED\n")
    return True


def test_benchmark_runtime():
    """Test optimizer runtime performance."""
    print("=" * 80)
    print("TEST 10: Runtime Benchmark")
    print("=" * 80)
    
    # Test with size 16
    optimizer_16 = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        size_budget=16,
    )
    
    bench_16 = optimizer_16.benchmark_runtime(num_runs=5)
    print(f"  Size 16: {bench_16['mean_ms']:.2f}ms (mean)")
    assert bench_16['mean_ms'] < 1000, \
        f"Runtime {bench_16['mean_ms']:.2f}ms >= 1000ms target"
    print("  Runtime < 1 second: PASS")
    
    # Test with size 32
    optimizer_32 = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        size_budget=32,
    )
    
    bench_32 = optimizer_32.benchmark_runtime(num_runs=5)
    print(f"  Size 32: {bench_32['mean_ms']:.2f}ms (mean)")
    assert bench_32['mean_ms'] < 1000, \
        f"Runtime {bench_32['mean_ms']:.2f}ms >= 1000ms target"
    print("  Runtime < 1 second: PASS")
    
    print("TEST 10: ALL PASSED\n")
    return True


def test_tree_sizes_4_8_16():
    """Test with various tree sizes."""
    print("=" * 80)
    print("TEST 11: Various Tree Sizes (4, 8, 16)")
    print("=" * 80)
    
    for size in [4, 8, 16]:
        optimizer = SequoiaTreeOptimizer(
            base_acceptance_rate=0.54,
            size_budget=size,
        )
        
        tree = optimizer.find_optimal_tree()
        
        assert tree.num_nodes <= size
        assert tree.expected_tokens > 0
        
        # Verify runtime
        bench = optimizer.benchmark_runtime(num_runs=3)
        assert bench['mean_ms'] < 1000
        
        print(f"  Size {size}: E[accepted]={tree.expected_tokens:.3f}, "
              f"runtime={bench['mean_ms']:.2f}ms: PASS")
    
    print("TEST 11: ALL PASSED\n")
    return True


def test_tree_validity():
    """Test that output tree is a valid DAG with correct parent-child relationships."""
    print("=" * 80)
    print("TEST 12: Tree Validity (DAG, Parent-Child)")
    print("=" * 80)
    
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        max_depth=4,
        max_branching=2,
        size_budget=16,
    )
    
    tree = optimizer.find_optimal_tree()
    
    if not tree.edges:
        print("  Single node tree (trivially valid): PASS")
        return True
    
    # Build parent map
    parent_map = {}
    children_map = {}
    for parent, child in tree.edges:
        if child in parent_map:
            raise AssertionError(f"Node {child} has multiple parents")
        parent_map[child] = parent
        
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)
    
    print("  No node has multiple parents: PASS")
    
    # Root should have no parent
    roots = [i for i in range(tree.num_nodes) if i not in parent_map]
    assert len(roots) == 1, f"Expected 1 root, found {len(roots)}"
    assert roots[0] == 0, f"Root should be node 0, found {roots[0]}"
    print("  Exactly one root (node 0): PASS")
    
    # Check branching factor
    for parent, children in children_map.items():
        assert len(children) <= optimizer.max_branching, \
            f"Node {parent} has {len(children)} children (max: {optimizer.max_branching})"
    print(f"  Branching factor <= {optimizer.max_branching}: PASS")
    
    # Check depth via BFS
    depths = {0: 0}
    queue = [0]
    while queue:
        node = queue.pop(0)
        node_depth = depths[node]
        assert node_depth <= optimizer.max_depth, \
            f"Node {node} at depth {node_depth} > max {optimizer.max_depth}"
        
        if node in children_map:
            for child in children_map[node]:
                depths[child] = node_depth + 1
                queue.append(child)
    
    print(f"  Max depth <= {optimizer.max_depth}: PASS")
    
    print("TEST 12: ALL PASSED\n")
    return True


def test_dp_recurrence_correctness():
    """Test DP recurrence formula manually."""
    print("=" * 80)
    print("TEST 13: DP Recurrence Correctness")
    print("=" * 80)
    
    # Simple case: p = 0.5, budget = 3
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.5,
        max_depth=2,
        max_branching=2,
        size_budget=3,
    )
    
    # Manual calculation:
    # Option 1: Chain 0->1->2
    #   E = 0.5 * (1 + 0.5 * (1 + 0.5)) = 0.5 * (1 + 0.75) = 0.875
    # Option 2: Root with 2 children (0->1, 0->2)
    #   E = 0.5 * (1 + 0.5 + 0.5) = 1.0
    # Optimal: Option 2
    
    tree = optimizer.find_optimal_tree()
    
    print(f"  Optimal E[accepted]: {tree.expected_tokens:.4f}")
    print(f"  Expected (star): 1.0")
    print(f"  Chain would give: 0.875")
    
    # Should choose star over chain
    assert abs(tree.expected_tokens - 1.0) < 1e-6, \
        f"Expected 1.0 (star), got {tree.expected_tokens:.4f}"
    
    # Verify tree structure is star (root with 2 children)
    assert len(tree.edges) == 2
    assert tree.edges[0][0] == 0  # Both edges from root
    assert tree.edges[1][0] == 0
    print("  Tree structure is star (optimal): PASS")
    
    print("TEST 13: ALL PASSED\n")
    return True


def run_all_tests():
    """Run all tree optimizer tests."""
    print("\n" + "=" * 80)
    print("TREE OPTIMIZER TEST SUITE")
    print("=" * 80 + "\n")
    
    results = {
        "Initialization": test_optimizer_initialization(),
        "Acceptance Probs": test_acceptance_probabilities(),
        "Subtree Expectation": test_subtree_expectation(),
        "DP Solver": test_dp_solver(),
        "TreeTopology": test_tree_topology(),
        "Find Optimal Tree": test_find_optimal_tree(),
        "Expected Tokens Target": test_expected_tokens_target(),
        "Position-Dependent": test_position_dependent_optimizer(),
        "Compare with Chain": test_compare_with_chain(),
        "Runtime Benchmark": test_benchmark_runtime(),
        "Various Tree Sizes": test_tree_sizes_4_8_16(),
        "Tree Validity": test_tree_validity(),
        "DP Recurrence": test_dp_recurrence_correctness(),
    }
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{total - passed} TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
