"""
Sequoia-style Dynamic Programming Tree Optimizer.

This module implements the DP algorithm from Sequoia to find the optimal
verification tree structure that maximizes E[accepted_tokens] / verification_cost.

Key Insights:
- Verification cost is dominated by 64 allreduce calls (constant ~2.94ms)
- Tree topology affects expected accepted tokens, not verification cost
- DP finds the tree structure maximizing E[accepted_tokens] for a given budget

Parameters:
- Per-position acceptance probability: 54% overall (can decay with depth)
- Hardware cost: 64 allreduces × 46µs = 2.94ms (constant per verification step)
- Tree constraints: max depth D, max branching factor B, total size <= budget

DP Recurrence:
For each subtree rooted at depth d:
  E[tokens] = acceptance_prob(d) * (1 + sum(E[child_subtrees]))

The algorithm enumerates tree topologies up to the size budget and selects
the one maximizing E[accepted_tokens].
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TreeTopology:
    """Represents a tree topology with parent-child relationships.
    
    Attributes:
        edges: List of (parent, child) tuples defining the tree structure
        num_nodes: Total number of nodes in the tree
        expected_tokens: Expected number of accepted tokens
    """
    edges: List[Tuple[int, int]]
    num_nodes: int
    expected_tokens: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'edges': self.edges,
            'num_nodes': self.num_nodes,
            'expected_tokens': self.expected_tokens,
        }


class SequoiaTreeOptimizer:
    """Dynamic programming optimizer for tree-structured speculative decoding.
    
    This optimizer finds the tree topology that maximizes expected accepted tokens
    per verification step, given hardware cost constraints and acceptance rates.
    
    The key insight: verification cost (64 allreduces) is constant regardless of
    tree size (up to GPU memory limits), so we want to maximize E[tokens] for
    a given size budget.
    
    Example:
        >>> optimizer = SequoiaTreeOptimizer(
        ...     base_acceptance_rate=0.54,
        ...     max_depth=4,
        ...     max_branching=2,
        ...     size_budget=16
        ... )
        >>> best_tree = optimizer.find_optimal_tree()
        >>> print(f"Expected tokens: {best_tree.expected_tokens:.2f}")
        >>> print(f"Tree edges: {best_tree.edges}")
    """
    
    def __init__(
        self,
        base_acceptance_rate: float = 0.54,
        max_depth: int = 4,
        max_branching: int = 2,
        size_budget: int = 16,
        acceptance_decay: float = 0.0,
        allreduce_latency_us: float = 46.0,
        num_allreduces: int = 64,
    ):
        """Initialize the tree optimizer.
        
        Args:
            base_acceptance_rate: Base token acceptance probability (default 0.54)
            max_depth: Maximum tree depth (root = depth 0)
            max_branching: Maximum branching factor per node
            size_budget: Maximum total tree size (nodes)
            acceptance_decay: Per-depth acceptance decay rate (default 0.0 = uniform)
                             If > 0, acceptance at depth d = base_rate * (1 - decay)^d
            allreduce_latency_us: Allreduce latency in microseconds
            num_allreduces: Number of allreduce calls per verification step
        """
        self.base_acceptance_rate = base_acceptance_rate
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.size_budget = size_budget
        self.acceptance_decay = acceptance_decay
        self.allreduce_latency_us = allreduce_latency_us
        self.num_allreduces = num_allreduces
        
        # DP cache: (remaining_budget, depth) -> (expected_tokens, best_children_config)
        self._dp_cache: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}
        
        # Precompute acceptance probabilities for each depth
        self._acceptance_probs = self._compute_acceptance_probs()
        
        # Hardware cost (constant for all trees up to memory limits)
        self.verification_cost_ms = (num_allreduces * allreduce_latency_us) / 1000.0
    
    def _compute_acceptance_probs(self) -> List[float]:
        """Compute acceptance probability for each depth.
        
        Returns:
            List of acceptance probabilities, indexed by depth
        """
        probs = []
        for d in range(self.max_depth + 1):
            if self.acceptance_decay > 0:
                # Exponential decay: p(d) = base * (1 - decay)^d
                prob = self.base_acceptance_rate * ((1 - self.acceptance_decay) ** d)
            else:
                # Uniform acceptance
                prob = self.base_acceptance_rate
            probs.append(prob)
        return probs
    
    def get_acceptance_prob(self, depth: int) -> float:
        """Get acceptance probability for a given depth.
        
        Args:
            depth: Tree depth (0 = root)
            
        Returns:
            Acceptance probability at that depth
        """
        if depth >= len(self._acceptance_probs):
            return 0.0
        return self._acceptance_probs[depth]
    
    def _compute_subtree_expectation(
        self,
        root_depth: int,
        children_expectations: List[float]
    ) -> float:
        """Compute expected tokens for a subtree.
        
        In Sequoia-style tree verification:
        - All nodes in the tree are verified in ONE forward pass
        - A node is accepted if it and all its ancestors are correct
        - E[accepted] = sum over all nodes of P(path to node is correct)
        
        DP recurrence:
          For a node at depth d with children c1...cB:
          E[node accepted] = p(d)
          E[subtree] = p(depth) + sum(E[child_subtree] * p(depth))
                     = p(depth) * (1 + sum(E[child_subtrees]))
        
        This captures: if the root is accepted (prob p), we get the root plus
        whatever we get from children subtrees.
        
        Args:
            root_depth: Depth of the subtree root
            children_expectations: List of E[tokens] for each child subtree
            
        Returns:
            Expected tokens for this subtree
        """
        p = self.get_acceptance_prob(root_depth)
        children_sum = sum(children_expectations)
        return p * (1.0 + children_sum)
    
    def _solve_dp(self, remaining_budget: int, depth: int) -> Tuple[float, List[int]]:
        """Solve DP for subtree with given budget at given depth.
        
        Returns the maximum expected tokens and the optimal number of children
        for each child subtree.
        
        Args:
            remaining_budget: Number of nodes available for this subtree (including root)
            depth: Depth of the subtree root
            
        Returns:
            Tuple of (expected_tokens, children_subtree_sizes)
        """
        # Check cache
        key = (remaining_budget, depth)
        if key in self._dp_cache:
            return self._dp_cache[key]
        
        # Base cases
        if remaining_budget < 1:
            return (0.0, [])
        
        if depth > self.max_depth:
            return (0.0, [])
        
        # Root node uses 1 budget
        if remaining_budget == 1:
            # Leaf node: E = p(depth) * 1
            p = self.get_acceptance_prob(depth)
            self._dp_cache[key] = (p, [])
            return (p, [])
        
        # Try all possible branching configurations
        p = self.get_acceptance_prob(depth)
        best_expectation = p  # At minimum, just the root
        best_children_sizes = []
        
        # Remaining budget after accounting for root
        child_budget = remaining_budget - 1
        
        # Try different numbers of children (0 to max_branching)
        for num_children in range(1, min(self.max_branching, child_budget) + 1):
            # Distribute child_budget among num_children
            # Try different distributions
            for child_sizes in self._generate_child_distributions(child_budget, num_children):
                # Compute expected tokens for this configuration
                children_expectations = []
                for child_size in child_sizes:
                    child_exp, _ = self._solve_dp(child_size, depth + 1)
                    children_expectations.append(child_exp)
                
                # Total expectation for this subtree
                total_exp = self._compute_subtree_expectation(depth, children_expectations)
                
                if total_exp > best_expectation:
                    best_expectation = total_exp
                    best_children_sizes = list(child_sizes)
        
        self._dp_cache[key] = (best_expectation, best_children_sizes)
        return (best_expectation, best_children_sizes)
    
    def _generate_child_distributions(
        self,
        total_budget: int,
        num_children: int
    ):
        """Generate all ways to distribute budget among children.
        
        Each child must have at least 1 node, and at most (total_budget - num_children + 1).
        
        Args:
            total_budget: Total nodes to distribute
            num_children: Number of children
            
        Yields:
            Tuples of child budget allocations
        """
        if num_children == 1:
            yield (total_budget,)
            return
        
        # First child can have 1 to (total_budget - num_children + 1) nodes
        for first in range(1, total_budget - num_children + 2):
            remaining = total_budget - first
            for rest in self._generate_child_distributions(remaining, num_children - 1):
                yield (first,) + rest
    
    def _reconstruct_tree(
        self,
        budget: int,
        depth: int,
        node_offset: int = 0
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Reconstruct the optimal tree from DP solution.
        
        Args:
            budget: Budget for this subtree
            depth: Depth of subtree root
            node_offset: Starting node index
            
        Returns:
            Tuple of (edges, next_available_node_index)
        """
        if budget < 1 or depth > self.max_depth:
            return ([], node_offset)
        
        # Root of this subtree
        root_idx = node_offset
        next_idx = node_offset + 1
        
        if budget == 1:
            # Leaf node
            return ([], next_idx)
        
        # Get optimal children configuration from DP
        _, children_sizes = self._dp_cache.get((budget, depth), (0.0, []))
        
        edges = []
        current_budget = budget - 1  # Account for root
        
        for child_size in children_sizes:
            # Recursively build child subtree
            child_edges, next_idx = self._reconstruct_tree(
                child_size, depth + 1, next_idx
            )
            
            # Add edge from root to child
            edges.append((root_idx, next_idx - child_size))
            edges.extend(child_edges)
            
            current_budget -= child_size
        
        return (edges, next_idx)
    
    def find_optimal_tree(self) -> TreeTopology:
        """Find the optimal tree topology.
        
        Returns:
            TreeTopology with optimal structure
        """
        # Clear cache for fresh computation
        self._dp_cache = {}
        
        # Solve DP
        expected_tokens, _ = self._solve_dp(self.size_budget, 0)
        
        # Reconstruct tree
        edges, _ = self._reconstruct_tree(self.size_budget, 0)
        
        # Count actual nodes (may be less than budget if optimal)
        if edges:
            num_nodes = max(max(e) for e in edges) + 1
        else:
            num_nodes = 1  # Just root
        
        return TreeTopology(
            edges=edges,
            num_nodes=num_nodes,
            expected_tokens=expected_tokens,
        )
    
    def compare_with_chain(self) -> Dict:
        """Compare optimal tree against flat (linear chain) speculation.
        
        A chain of length L has:
          E[chain] = sum of P(path to node i is correct) for i in 0..L-1
          P(path to node i is correct) = product of p(d) for d in 0..i
        
        For uniform p: E[chain] = p + p^2 + p^3 + ... + p^L
                     = p * (1 - p^L) / (1 - p)  [geometric series]
        
        This is because in chain speculation, each token must be verified
        sequentially, and node i is only accepted if ALL previous nodes are correct.
        
        Returns:
            Dict with comparison metrics
        """
        # Find optimal tree
        optimal_tree = self.find_optimal_tree()
        
        # Compute chain expectation for same size
        chain_length = optimal_tree.num_nodes
        
        # Chain expectation: sum of path probabilities
        chain_expectation = 0.0
        path_prob = 1.0
        for d in range(chain_length):
            p = self.get_acceptance_prob(d)
            if p <= 0:
                break  # Can't extend beyond this depth
            path_prob *= p
            chain_expectation += path_prob
        
        # Compute efficiency (tokens per node)
        tree_efficiency = optimal_tree.expected_tokens / optimal_tree.num_nodes
        chain_efficiency = chain_expectation / chain_length
        
        return {
            'optimal_tree': optimal_tree.to_dict(),
            'chain': {
                'length': chain_length,
                'expected_tokens': chain_expectation,
            },
            'improvement': {
                'absolute': optimal_tree.expected_tokens - chain_expectation,
                'relative': (optimal_tree.expected_tokens / chain_expectation - 1) * 100
                    if chain_expectation > 0 else 0,
            },
            'efficiency': {
                'tree': tree_efficiency,
                'chain': chain_efficiency,
            }
        }
    
    def benchmark_runtime(self, num_runs: int = 10) -> Dict:
        """Benchmark optimizer runtime.
        
        Args:
            num_runs: Number of runs to average
            
        Returns:
            Dict with timing statistics
        """
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.find_optimal_tree()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return {
            'mean_ms': sum(times) / len(times) * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
            'num_runs': num_runs,
        }


def create_position_dependent_optimizer(
    code_domain: bool = False,
    repetitive_domain: bool = False,
) -> SequoiaTreeOptimizer:
    """Create an optimizer with position-dependent acceptance rates.
    
    Based on RESEARCH.md measurements:
    - Overall: 54% acceptance
    - Code: 59% acceptance
    - Repetitive: 87% acceptance
    - JSON: 39% acceptance
    - Conversational: 33% acceptance
    
    Args:
        code_domain: Use code-domain acceptance rates
        repetitive_domain: Use repetitive-domain acceptance rates
        
    Returns:
        Configured SequoiaTreeOptimizer
    """
    if repetitive_domain:
        # Higher base rate, moderate decay
        return SequoiaTreeOptimizer(
            base_acceptance_rate=0.87,
            max_depth=4,
            max_branching=3,
            size_budget=16,
            acceptance_decay=0.05,
        )
    elif code_domain:
        # Moderate base rate, mild decay
        return SequoiaTreeOptimizer(
            base_acceptance_rate=0.59,
            max_depth=4,
            max_branching=2,
            size_budget=16,
            acceptance_decay=0.05,
        )
    else:
        # Default (overall)
        return SequoiaTreeOptimizer(
            base_acceptance_rate=0.54,
            max_depth=4,
            max_branching=2,
            size_budget=16,
            acceptance_decay=0.0,
        )


# ============================================================================
# Main (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Sequoia Tree Optimizer - Demonstration")
    print("=" * 80)
    
    # Test 1: Uniform acceptance (54%)
    print("\n1. Uniform Acceptance (54%)")
    print("-" * 40)
    optimizer = SequoiaTreeOptimizer(
        base_acceptance_rate=0.54,
        max_depth=4,
        max_branching=2,
        size_budget=16,
    )
    
    tree = optimizer.find_optimal_tree()
    print(f"Optimal tree:")
    print(f"  Nodes: {tree.num_nodes}")
    print(f"  Edges: {tree.edges}")
    print(f"  E[accepted]: {tree.expected_tokens:.3f}")
    print(f"  Verification cost: {optimizer.verification_cost_ms:.2f}ms")
    
    # Compare with chain
    comparison = optimizer.compare_with_chain()
    print(f"\nComparison with flat chain:")
    print(f"  Chain E[accepted]: {comparison['chain']['expected_tokens']:.3f}")
    print(f"  Tree improvement: {comparison['improvement']['relative']:.1f}%")
    
    # Benchmark
    bench = optimizer.benchmark_runtime()
    print(f"\nRuntime: {bench['mean_ms']:.2f}ms (mean)")
    
    # Test 2: Position-dependent acceptance (code domain)
    print("\n\n2. Code Domain (59% base, 5% decay)")
    print("-" * 40)
    optimizer_code = create_position_dependent_optimizer(code_domain=True)
    tree_code = optimizer_code.find_optimal_tree()
    print(f"Optimal tree:")
    print(f"  Nodes: {tree_code.num_nodes}")
    print(f"  E[accepted]: {tree_code.expected_tokens:.3f}")
    
    # Test 3: High acceptance (repetitive domain)
    print("\n\n3. Repetitive Domain (87% base, 5% decay)")
    print("-" * 40)
    optimizer_rep = create_position_dependent_optimizer(repetitive_domain=True)
    tree_rep = optimizer_rep.find_optimal_tree()
    print(f"Optimal tree:")
    print(f"  Nodes: {tree_rep.num_nodes}")
    print(f"  E[accepted]: {tree_rep.expected_tokens:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Uniform (54%):    E[accepted] = {tree.expected_tokens:.3f} (target: >= 2.0)")
    print(f"Code (59%):       E[accepted] = {tree_code.expected_tokens:.3f}")
    print(f"Repetitive (87%): E[accepted] = {tree_rep.expected_tokens:.3f}")
    
    if tree.expected_tokens >= 2.0:
        print("\n✓ Target achieved: E[accepted] >= 2.0 at 54% acceptance")
    else:
        print("\n✗ Target NOT achieved: E[accepted] < 2.0 at 54% acceptance")
    
    if bench['mean_ms'] < 1000:
        print("✓ Runtime target achieved: < 1 second for tree size <= 16")
    else:
        print("✗ Runtime target NOT achieved: >= 1 second")
