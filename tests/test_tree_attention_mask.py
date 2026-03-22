#!/usr/bin/env python3
"""
Tree Attention Mask Tests.

This test suite validates the TreeAttentionMask implementation for
Sequoia-style speculative decoding.

Tests cover:
1. Tree topology construction (binary, chain, star trees)
2. Mask correctness (ancestors, self, KV cache)
3. Various tree sizes and depths
4. Edge cases (empty KV, single node, etc.)
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from inference.tree_attention import (
    TreeNode,
    TreeTopology,
    TreeAttentionMask,
    build_complete_binary_tree,
    build_chain_tree,
    build_star_tree,
    verify_tree_mask_correctness
)


def test_tree_node_basic():
    """Test basic TreeNode functionality."""
    print("=" * 80)
    print("TEST 1: TreeNode Basic Functionality")
    print("=" * 80)
    
    # Create root
    root = TreeNode(token_id=100)
    assert root.token_id == 100
    assert root.parent is None
    assert root.depth == 0
    assert root.children == []
    print("  Root node created: PASS")
    
    # Create children
    child1 = TreeNode(token_id=200, parent=root)
    child2 = TreeNode(token_id=300, parent=root)
    
    assert len(root.children) == 2
    assert child1.depth == 1
    assert child2.depth == 1
    assert child1.parent == root
    print("  Child nodes created: PASS")
    
    # Test ancestors
    ancestors = child1.get_ancestors()
    assert len(ancestors) == 1
    assert ancestors[0] == root
    print("  Ancestors retrieval: PASS")
    
    # Test path from root
    path = child1.get_path_from_root()
    assert len(path) == 2
    assert path[0] == root
    assert path[1] == child1
    print("  Path from root: PASS")
    
    # Create grandchild
    grandchild = TreeNode(token_id=400, parent=child1)
    ancestors = grandchild.get_ancestors()
    assert len(ancestors) == 2
    assert ancestors[0] == root
    assert ancestors[1] == child1
    print("  Grandchild ancestors: PASS")
    
    print("TEST 1: ALL PASSED\n")
    return True


def test_tree_topology_basic():
    """Test TreeTopology construction and traversal."""
    print("=" * 80)
    print("TEST 2: TreeTopology Basic Functionality")
    print("=" * 80)
    
    # Create tree
    tree = TreeTopology(root_token=100)
    root = tree.root
    
    # Add children
    child1 = tree.add_child(root, 200)
    child2 = tree.add_child(root, 300)
    grandchild = tree.add_child(child1, 400)
    
    # Test node count
    assert tree.get_tree_size() == 4
    print(f"  Tree size: {tree.get_tree_size()} (expected 4): PASS")
    
    # Test max depth
    assert tree.get_max_depth() == 2
    print(f"  Max depth: {tree.get_max_depth()} (expected 2): PASS")
    
    # Test node indices (BFS order)
    assert root.index == 0
    assert child1.index == 1
    assert child2.index == 2
    assert grandchild.index == 3
    print("  Node indices (BFS): PASS")
    
    # Test get_all_tokens
    tokens = tree.get_all_tokens()
    assert tokens == [100, 200, 300, 400]
    print(f"  Token IDs: {tokens}: PASS")
    
    # Test get_nodes_at_depth
    depth0 = tree.get_nodes_at_depth(0)
    depth1 = tree.get_nodes_at_depth(1)
    depth2 = tree.get_nodes_at_depth(2)
    
    assert len(depth0) == 1 and depth0[0] == root
    assert len(depth1) == 2
    assert len(depth2) == 1 and depth2[0] == grandchild
    print("  Nodes at depth: PASS")
    
    # Test serialization
    tree_dict = tree.to_dict()
    assert tree_dict['tree_size'] == 4
    assert tree_dict['max_depth'] == 2
    assert len(tree_dict['nodes']) == 4
    print("  Tree serialization: PASS")
    
    print("TEST 2: ALL PASSED\n")
    return True


def test_build_complete_binary_tree():
    """Test complete binary tree construction."""
    print("=" * 80)
    print("TEST 3: Complete Binary Tree Construction")
    print("=" * 80)
    
    # Test depth 0 (only root)
    tree0 = build_complete_binary_tree(root_token=100, depth=0)
    assert tree0.get_tree_size() == 1
    assert tree0.get_max_depth() == 0
    print("  Depth 0 (root only): PASS")
    
    # Test depth 1 (root + 2 children)
    tree1 = build_complete_binary_tree(root_token=100, depth=1)
    assert tree1.get_tree_size() == 3
    assert tree1.get_max_depth() == 1
    tokens = tree1.get_all_tokens()
    assert tokens == [100, 101, 102]
    print(f"  Depth 1: size={tree1.get_tree_size()}, tokens={tokens}: PASS")
    
    # Test depth 2 (root + 2 children + 4 grandchildren)
    tree2 = build_complete_binary_tree(root_token=100, depth=2)
    assert tree2.get_tree_size() == 7
    assert tree2.get_max_depth() == 2
    tokens = tree2.get_all_tokens()
    assert tokens == [100, 101, 102, 103, 104, 105, 106]
    print(f"  Depth 2: size={tree2.get_tree_size()}, tokens={tokens}: PASS")
    
    # Test depth 3
    tree3 = build_complete_binary_tree(root_token=100, depth=3)
    assert tree3.get_tree_size() == 15
    assert tree3.get_max_depth() == 3
    print(f"  Depth 3: size={tree3.get_tree_size()}: PASS")
    
    # Test with custom token IDs
    custom_tokens = [10, 20, 30, 40, 50, 60, 70]
    tree_custom = build_complete_binary_tree(root_token=10, depth=2, token_ids=custom_tokens)
    assert tree_custom.get_all_tokens() == custom_tokens
    print(f"  Custom token IDs: {tree_custom.get_all_tokens()}: PASS")
    
    print("TEST 3: ALL PASSED\n")
    return True


def test_build_chain_tree():
    """Test chain (linear) tree construction."""
    print("=" * 80)
    print("TEST 4: Chain Tree Construction")
    print("=" * 80)
    
    # Test simple chain
    tokens = [100, 200, 300, 400]
    tree = build_chain_tree(tokens)
    
    assert tree.get_tree_size() == 4
    assert tree.get_max_depth() == 3
    assert tree.get_all_tokens() == tokens
    print(f"  Chain of 4: size={tree.get_tree_size()}, depth={tree.get_max_depth()}: PASS")
    
    # Verify parent-child relationships
    root = tree.root
    assert root.token_id == 100
    assert len(root.children) == 1
    assert root.children[0].token_id == 200
    
    child1 = root.children[0]
    assert len(child1.children) == 1
    assert child1.children[0].token_id == 300
    print("  Parent-child chain: PASS")
    
    # Test single node chain
    single_tree = build_chain_tree([999])
    assert single_tree.get_tree_size() == 1
    assert single_tree.get_max_depth() == 0
    print("  Single node chain: PASS")
    
    print("TEST 4: ALL PASSED\n")
    return True


def test_build_star_tree():
    """Test star-shaped tree construction."""
    print("=" * 80)
    print("TEST 5: Star Tree Construction")
    print("=" * 80)
    
    # Test star with 3 children
    root_token = 100
    children = [200, 300, 400]
    tree = build_star_tree(root_token, children)
    
    assert tree.get_tree_size() == 4
    assert tree.get_max_depth() == 1
    print(f"  Star (1 root + 3 children): size={tree.get_tree_size()}, depth={tree.get_max_depth()}: PASS")
    
    # Verify structure
    assert tree.root.token_id == root_token
    assert len(tree.root.children) == 3
    child_tokens = [c.token_id for c in tree.root.children]
    assert child_tokens == children
    print(f"  Children tokens: {child_tokens}: PASS")
    
    # All children should have no children themselves
    for child in tree.root.children:
        assert len(child.children) == 0
    print("  No grandchildren: PASS")
    
    print("TEST 5: ALL PASSED\n")
    return True


def test_tree_attention_mask_basic():
    """Test basic tree attention mask construction."""
    print("=" * 80)
    print("TEST 6: Tree Attention Mask - Basic")
    print("=" * 80)
    
    # Create a simple tree
    tree = build_complete_binary_tree(root_token=100, depth=1)
    # Tree structure:
    #     0 (root)
    #     ├── 1 (left child)
    #     └── 2 (right child)
    
    # Test without KV cache
    mask_no_kv = TreeAttentionMask(tree, kv_len=0)
    assert mask_no_kv.tree_size == 3
    assert mask_no_kv.kv_len == 0
    assert mask_no_kv.mask.shape == (3, 3)
    print(f"  Mask shape (no KV): {mask_no_kv.mask.shape}: PASS")
    
    # Test with KV cache
    mask_with_kv = TreeAttentionMask(tree, kv_len=2)
    assert mask_with_kv.kv_len == 2
    assert mask_with_kv.mask.shape == (3, 5)  # 3 tree + 2 KV
    print(f"  Mask shape (with KV): {mask_with_kv.mask.shape}: PASS")
    
    # Verify mask values
    dense = mask_with_kv.to_dense_int()
    
    # Row 0 (root): attends to self + KV
    expected_row0 = [1, 0, 0, 1, 1]  # self + 2 KV
    assert list(dense[0]) == expected_row0
    print(f"  Row 0 (root): {list(dense[0])}: PASS")
    
    # Row 1 (left child): attends to root, self + KV
    expected_row1 = [1, 1, 0, 1, 1]  # root + self + 2 KV
    assert list(dense[1]) == expected_row1
    print(f"  Row 1 (left child): {list(dense[1])}: PASS")
    
    # Row 2 (right child): attends to root, self + KV
    expected_row2 = [1, 0, 1, 1, 1]  # root + self + 2 KV
    assert list(dense[2]) == expected_row2
    print(f"  Row 2 (right child): {list(dense[2])}: PASS")
    
    print("TEST 6: ALL PASSED\n")
    return True


def test_tree_attention_mask_deeper():
    """Test tree attention mask for deeper trees."""
    print("=" * 80)
    print("TEST 7: Tree Attention Mask - Deeper Tree")
    print("=" * 80)
    
    # Create depth-2 tree (7 nodes)
    tree = build_complete_binary_tree(root_token=100, depth=2)
    # Tree structure:
    #         0
    #     /       \
    #    1         2
    #   / \       / \
    #  3   4     5   6
    
    mask = TreeAttentionMask(tree, kv_len=2)
    dense = mask.to_dense_int()
    
    assert mask.tree_size == 7
    assert mask.mask.shape == (7, 9)
    print(f"  Mask shape: {mask.mask.shape}: PASS")
    
    # Verify specific attention patterns
    
    # Node 0 (root): attends to self + KV
    assert dense[0, 0] == 1
    assert dense[0, 1:7].sum() == 0  # No other tree nodes
    assert dense[0, 7:9].sum() == 2  # All KV
    print("  Root attention: PASS")
    
    # Node 3 (grandchild of root via left child)
    # Attends to: root(0), left_child(1), self(3), KV
    assert dense[3, 0] == 1  # root
    assert dense[3, 1] == 1  # parent (node 1)
    assert dense[3, 2] == 0  # uncle (node 2) - NOT attended
    assert dense[3, 3] == 1  # self
    assert dense[3, 4:7].sum() == 0  # cousins - NOT attended
    assert dense[3, 7:9].sum() == 2  # All KV
    print("  Grandchild attention: PASS")
    
    print("TEST 7: ALL PASSED\n")
    return True


def test_tree_attention_mask_chain():
    """Test tree attention mask for chain topology."""
    print("=" * 80)
    print("TEST 8: Tree Attention Mask - Chain Tree")
    print("=" * 80)
    
    # Create chain: 0 -> 1 -> 2 -> 3
    tree = build_chain_tree([100, 200, 300, 400])
    mask = TreeAttentionMask(tree, kv_len=1)
    dense = mask.to_dense_int()
    
    # Chain should have lower triangular mask (causal)
    # Each node attends to all previous nodes + self + KV
    
    # Node 0: attends to self + KV
    assert list(dense[0]) == [1, 0, 0, 0, 1]
    
    # Node 1: attends to 0, self + KV
    assert list(dense[1]) == [1, 1, 0, 0, 1]
    
    # Node 2: attends to 0, 1, self + KV
    assert list(dense[2]) == [1, 1, 1, 0, 1]
    
    # Node 3: attends to 0, 1, 2, self + KV
    assert list(dense[3]) == [1, 1, 1, 1, 1]
    
    print("  Chain mask (lower triangular): PASS")
    print("TEST 8: ALL PASSED\n")
    return True


def test_tree_attention_mask_correctness():
    """Test comprehensive mask correctness verification."""
    print("=" * 80)
    print("TEST 9: Tree Attention Mask Correctness Verification")
    print("=" * 80)
    
    # Test binary tree
    tree_binary = build_complete_binary_tree(root_token=100, depth=2)
    mask_binary = TreeAttentionMask(tree_binary, kv_len=5)
    assert verify_tree_mask_correctness(mask_binary, verbose=True)
    print("  Binary tree mask: PASS")
    
    # Test chain tree
    tree_chain = build_chain_tree([1, 2, 3, 4, 5])
    mask_chain = TreeAttentionMask(tree_chain, kv_len=3)
    assert verify_tree_mask_correctness(mask_chain, verbose=True)
    print("  Chain tree mask: PASS")
    
    # Test star tree
    tree_star = build_star_tree(root_token=100, children_tokens=[200, 300, 400, 500])
    mask_star = TreeAttentionMask(tree_star, kv_len=2)
    assert verify_tree_mask_correctness(mask_star, verbose=True)
    print("  Star tree mask: PASS")
    
    # Test larger tree
    tree_large = build_complete_binary_tree(root_token=100, depth=3)
    mask_large = TreeAttentionMask(tree_large, kv_len=10)
    assert verify_tree_mask_correctness(mask_large, verbose=True)
    print("  Large tree (depth 3) mask: PASS")
    
    print("TEST 9: ALL PASSED\n")
    return True


def test_tree_attention_mask_edge_cases():
    """Test edge cases."""
    print("=" * 80)
    print("TEST 10: Tree Attention Mask - Edge Cases")
    print("=" * 80)
    
    # Single node tree, no KV
    tree_single = build_chain_tree([100])
    mask_single = TreeAttentionMask(tree_single, kv_len=0)
    assert mask_single.mask.shape == (1, 1)
    assert mask_single.mask[0, 0] == True
    print("  Single node, no KV: PASS")
    
    # Single node tree, with KV
    mask_single_kv = TreeAttentionMask(tree_single, kv_len=5)
    assert mask_single_kv.mask.shape == (1, 6)
    assert mask_single_kv.mask[0, 0] == True
    assert mask_single_kv.mask[0, 1:].all()  # All KV attended
    print("  Single node, with KV: PASS")
    
    # Zero KV length
    tree = build_complete_binary_tree(root_token=100, depth=1)
    mask_zero_kv = TreeAttentionMask(tree, kv_len=0)
    assert mask_zero_kv.kv_len == 0
    assert mask_zero_kv.mask.shape == (3, 3)
    # Should only attend to tree ancestors
    assert not mask_zero_kv.mask[:, 3:].any()  # No columns beyond tree
    print("  Zero KV length: PASS")
    
    # Update KV length
    mask_zero_kv.update_kv_len(3)
    assert mask_zero_kv.mask.shape == (3, 6)
    assert mask_zero_kv.kv_len == 3
    print("  Update KV length: PASS")
    
    print("TEST 10: ALL PASSED\n")
    return True


def test_visualization():
    """Test mask visualization."""
    print("=" * 80)
    print("TEST 11: Tree Attention Mask Visualization")
    print("=" * 80)
    
    tree = build_complete_binary_tree(root_token=100, depth=1)
    mask = TreeAttentionMask(tree, kv_len=2)
    
    viz = mask.visualize()
    print("\n" + viz)
    
    assert "T 0" in viz
    assert "T 1" in viz
    assert "T 2" in viz
    assert "K 0" in viz
    assert "K 1" in viz
    print("\n  Visualization contains all labels: PASS")
    
    print("TEST 11: ALL PASSED\n")
    return True


def test_tree_sizes_4_8_16():
    """Test with tree sizes matching feature requirements (4, 8, 16)."""
    print("=" * 80)
    print("TEST 12: Tree Sizes 4, 8, 16 (Feature Requirements)")
    print("=" * 80)
    
    # Tree size 4: depth-2 binary tree has 7 nodes, so use depth-1 (3 nodes) + 1 more
    # Or use chain of 4
    tree_4 = build_chain_tree([1, 2, 3, 4])
    mask_4 = TreeAttentionMask(tree_4, kv_len=10)
    assert mask_4.tree_size == 4
    assert verify_tree_mask_correctness(mask_4)
    print(f"  Tree size 4: PASS (shape={mask_4.mask.shape})")
    
    # Tree size 8: need depth-2 binary tree (7 nodes) + 1, or custom tree
    # Use depth-3 binary tree and take first 8 nodes (won't be complete)
    # Better: build custom tree with 8 nodes
    tree_8 = build_complete_binary_tree(root_token=100, depth=2)  # 7 nodes
    # Add one more node to make it 8
    extra_node = tree_8.add_child(tree_8.root.children[0], 999)
    assert tree_8.get_tree_size() == 8
    mask_8 = TreeAttentionMask(tree_8, kv_len=10)
    assert mask_8.tree_size == 8
    assert verify_tree_mask_correctness(mask_8)
    print(f"  Tree size 8: PASS (shape={mask_8.mask.shape})")
    
    # Tree size 16: use depth-3 binary tree (15 nodes) + 1
    tree_16 = build_complete_binary_tree(root_token=100, depth=3)  # 15 nodes
    # Add one more node
    extra_node = tree_16.add_child(tree_16.root.children[0].children[0], 999)
    assert tree_16.get_tree_size() == 16
    mask_16 = TreeAttentionMask(tree_16, kv_len=10)
    assert mask_16.tree_size == 16
    assert verify_tree_mask_correctness(mask_16)
    print(f"  Tree size 16: PASS (shape={mask_16.mask.shape})")
    
    print("TEST 12: ALL PASSED\n")
    return True


def run_all_tests():
    """Run all tree attention mask tests."""
    print("\n" + "=" * 80)
    print("TREE ATTENTION MASK TEST SUITE")
    print("=" * 80 + "\n")
    
    results = {
        "TreeNode Basic": test_tree_node_basic(),
        "TreeTopology Basic": test_tree_topology_basic(),
        "Complete Binary Tree": test_build_complete_binary_tree(),
        "Chain Tree": test_build_chain_tree(),
        "Star Tree": test_build_star_tree(),
        "Mask Basic": test_tree_attention_mask_basic(),
        "Mask Deeper Tree": test_tree_attention_mask_deeper(),
        "Mask Chain": test_tree_attention_mask_chain(),
        "Mask Correctness": test_tree_attention_mask_correctness(),
        "Mask Edge Cases": test_tree_attention_mask_edge_cases(),
        "Visualization": test_visualization(),
        "Tree Sizes 4,8,16": test_tree_sizes_4_8_16(),
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
