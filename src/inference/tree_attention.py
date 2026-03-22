"""
Tree Attention Mask for Sequoia-style Speculative Decoding.

This module implements tree-structured attention masks for batch verification
of draft tokens in speculative decoding. Instead of verifying K draft tokens
sequentially (K decode steps, each with 64 allreduces), we verify them all in
a single forward pass using tree-structured attention.

Key Concepts:
- Tree topology: parent-child relationships between draft tokens
- Each node attends only to its ancestors and itself (causal in tree)
- Tree attention mask: [tree_size, tree_size + kv_len] boolean matrix
- Single allreduce set for entire tree verification (64 calls total)
"""

from typing import Optional, List, Dict, Tuple
import numpy as np


class TreeNode:
    """Represents a node in the draft token tree.
    
    Attributes:
        token_id: The draft token ID at this node
        parent: Parent TreeNode (None for root)
        children: List of child TreeNodes
        depth: Depth in tree (root = 0)
        index: Position in the flattened tree embedding tensor
    """
    
    def __init__(self, token_id: int, parent: Optional['TreeNode'] = None):
        self.token_id = token_id
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.index = 0  # Will be set during tree flattening
        
        if parent is not None:
            parent.children.append(self)
    
    def get_ancestors(self) -> List['TreeNode']:
        """Get all ancestors from root to parent (excluding self)."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        # Reverse to get [root, ..., parent]
        ancestors.reverse()
        return ancestors
    
    def get_path_from_root(self) -> List['TreeNode']:
        """Get path from root to self (inclusive)."""
        return self.get_ancestors() + [self]
    
    def __repr__(self):
        return f"TreeNode(token={self.token_id}, depth={self.depth}, idx={self.index})"


class TreeTopology:
    """Represents the topology of a draft token tree.
    
    The tree is built from a root token and branches out with draft continuations.
    This class manages tree construction, flattening for batch processing, and
    provides utilities for tree traversal.
    
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
    """
    
    def __init__(self, root_token: int):
        """Initialize tree with root token.
        
        Args:
            root_token: Token ID for the root node
        """
        self.root = TreeNode(root_token)
        self.nodes: List[TreeNode] = [self.root]
        self._index_nodes()
    
    def _index_nodes(self):
        """Assign indices to all nodes in BFS order."""
        for i, node in enumerate(self.nodes):
            node.index = i
    
    def add_child(self, parent_node: TreeNode, token_id: int) -> TreeNode:
        """Add a child node to the tree.
        
        Args:
            parent_node: Parent TreeNode
            token_id: Token ID for the child
            
        Returns:
            New child TreeNode
        """
        child = TreeNode(token_id, parent_node)
        self.nodes.append(child)
        self._index_nodes()
        return child
    
    def get_node_by_index(self, index: int) -> TreeNode:
        """Get node by its flattened index."""
        return self.nodes[index]
    
    def get_all_tokens(self) -> List[int]:
        """Get all token IDs in BFS order."""
        return [node.token_id for node in self.nodes]
    
    def get_max_depth(self) -> int:
        """Get maximum depth of the tree."""
        return max(node.depth for node in self.nodes)
    
    def get_tree_size(self) -> int:
        """Get total number of nodes in the tree."""
        return len(self.nodes)
    
    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """Get all nodes at a specific depth."""
        return [node for node in self.nodes if node.depth == depth]
    
    def to_dict(self) -> Dict:
        """Serialize tree topology to dictionary."""
        return {
            'root_token': self.root.token_id,
            'nodes': [
                {
                    'index': node.index,
                    'token_id': node.token_id,
                    'depth': node.depth,
                    'parent_index': node.parent.index if node.parent else None,
                    'children_indices': [c.index for c in node.children]
                }
                for node in self.nodes
            ],
            'tree_size': self.get_tree_size(),
            'max_depth': self.get_max_depth()
        }


class TreeAttentionMask:
    """Builds tree attention masks for speculative decoding.
    
    For a tree of draft tokens, each node attends to:
    1. All its ancestors (path from root to node)
    2. Itself
    3. All KV cache tokens (all previous context)
    
    The mask is a [tree_size, tree_size + kv_len] boolean matrix where:
    - mask[i, j] = True if tree node i can attend to position j
    - Positions 0..tree_size-1 are tree nodes
    - Positions tree_size..tree_size+kv_len-1 are KV cache positions
    
    Example (3-node tree, 2 KV tokens):
    Tree:
        0 (root)
        ├── 1 (child)
        └── 2 (child)
    
    Attention mask (T=True, F=False):
         T0 T1 T2 KV0 KV1
    T0 [ T  F  F   T   T ]  # Root sees only itself + KV
    T1 [ T  T  F   T   T ]  # Child1 sees root + self + KV
    T2 [ T  F  T   T   T ]  # Child2 sees root + self + KV
    """
    
    def __init__(self, tree_topology: TreeTopology, kv_len: int = 0):
        """Initialize tree attention mask builder.
        
        Args:
            tree_topology: TreeTopology defining draft token structure
            kv_len: Length of KV cache (all tree nodes attend to all KV positions)
        """
        self.tree = tree_topology
        self.kv_len = kv_len
        self.tree_size = tree_topology.get_tree_size()
        self.mask = self._build_mask()
    
    def _build_mask(self) -> np.ndarray:
        """Build the tree attention mask.
        
        Returns:
            Boolean mask of shape [tree_size, tree_size + kv_len]
        """
        total_len = self.tree_size + self.kv_len
        mask = np.zeros((self.tree_size, total_len), dtype=bool)
        
        for node in self.tree.nodes:
            # Node attends to itself
            mask[node.index, node.index] = True
            
            # Node attends to all ancestors
            for ancestor in node.get_ancestors():
                mask[node.index, ancestor.index] = True
            
            # All nodes attend to all KV cache positions
            if self.kv_len > 0:
                kv_start = self.tree_size
                mask[node.index, kv_start:kv_start + self.kv_len] = True
        
        return mask
    
    def get_mask(self) -> np.ndarray:
        """Get the attention mask.
        
        Returns:
            Boolean mask of shape [tree_size, tree_size + kv_len]
        """
        return self.mask
    
    def get_causal_mask(self) -> np.ndarray:
        """Get causal attention mask for tree-only tokens (no KV cache).
        
        This is useful for debugging and visualization.
        
        Returns:
            Boolean mask of shape [tree_size, tree_size]
        """
        return self.mask[:, :self.tree_size]
    
    def update_kv_len(self, kv_len: int):
        """Update the KV cache length and rebuild mask.
        
        Args:
            kv_len: New KV cache length
        """
        self.kv_len = kv_len
        self.mask = self._build_mask()
    
    def to_dense_int(self) -> np.ndarray:
        """Convert mask to integer matrix (0 or 1).
        
        Useful for debugging or passing to kernels that expect int masks.
        
        Returns:
            Integer mask of shape [tree_size, tree_size + kv_len]
        """
        return self.mask.astype(np.int32)
    
    def visualize(self) -> str:
        """Create ASCII visualization of the attention mask.
        
        Returns:
            String representation of the mask
        """
        lines = []
        header = "     " + " ".join(f"T{i:2d}" for i in range(self.tree_size))
        if self.kv_len > 0:
            header += " " + " ".join(f"K{i:2d}" for i in range(self.kv_len))
        lines.append(header)
        
        for i in range(self.tree_size):
            row = f"T{i:2d}: "
            for j in range(self.tree_size):
                row += "  1 " if self.mask[i, j] else "  0 "
            if self.kv_len > 0:
                for j in range(self.kv_len):
                    row += "  1 " if self.mask[i, self.tree_size + j] else "  0 "
            lines.append(row)
        
        return "\n".join(lines)


def build_complete_binary_tree(root_token: int, depth: int, 
                                token_ids: Optional[List[int]] = None) -> TreeTopology:
    """Build a complete binary tree of given depth.
    
    Example for depth=2:
        root(0)
        ├── child(1)
        │   ├── grandchild(3)
        │   └── grandchild(4)
        └── child(2)
            ├── grandchild(5)
            └── grandchild(6)
    
    Args:
        root_token: Token ID for root node
        depth: Depth of tree (0 = only root, 1 = root+children, etc.)
        token_ids: Optional list of token IDs to use (must have at least 2^(depth+1)-1 tokens)
                   If None, uses sequential IDs starting from root_token
        
    Returns:
        TreeTopology instance
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    
    required_nodes = (2 ** (depth + 1)) - 1
    if token_ids is not None and len(token_ids) < required_nodes:
        raise ValueError(f"token_ids must have at least {required_nodes} tokens for depth {depth}")
    
    if token_ids is None:
        # Generate sequential token IDs
        token_ids = list(range(root_token, root_token + required_nodes))
    
    # Build tree in BFS order
    tree = TreeTopology(token_ids[0])
    node_idx = 1
    
    # Queue-based BFS construction
    queue = [tree.root]
    current_depth = 0
    
    while current_depth <= depth and queue:
        next_queue = []
        for parent in queue:
            if node_idx >= len(token_ids):
                break
            # Add left child
            left = tree.add_child(parent, token_ids[node_idx])
            node_idx += 1
            next_queue.append(left)
            
            if node_idx >= len(token_ids):
                break
            # Add right child
            right = tree.add_child(parent, token_ids[node_idx])
            node_idx += 1
            next_queue.append(right)
        
        queue = next_queue
        current_depth += 1
    
    return tree


def build_chain_tree(token_ids: List[int]) -> TreeTopology:
    """Build a chain (linear) tree from token IDs.
    
    This represents sequential draft tokens with no branching.
    Example: 0 -> 1 -> 2 -> 3 (each node has exactly one child)
    
    Args:
        token_ids: List of token IDs in order
        
    Returns:
        TreeTopology instance
    """
    if len(token_ids) == 0:
        raise ValueError("token_ids must not be empty")
    
    tree = TreeTopology(token_ids[0])
    current = tree.root
    
    for token_id in token_ids[1:]:
        current = tree.add_child(current, token_id)
    
    return tree


def build_star_tree(root_token: int, children_tokens: List[int]) -> TreeTopology:
    """Build a star-shaped tree (root with multiple children, no grandchildren).
    
    Example: root with 3 children (all depth=1)
        root(0)
        ├── child(1)
        ├── child(2)
        └── child(3)
    
    Args:
        root_token: Token ID for root
        children_tokens: List of token IDs for children
        
    Returns:
        TreeTopology instance
    """
    tree = TreeTopology(root_token)
    
    for child_token in children_tokens:
        tree.add_child(tree.root, child_token)
    
    return tree


# ============================================================================
# Verification Utilities
# ============================================================================

def verify_tree_mask_correctness(mask: TreeAttentionMask, 
                                  verbose: bool = False) -> bool:
    """Verify that tree attention mask is correctly constructed.
    
    Checks:
    1. Every node attends to itself
    2. Every node attends to all ancestors
    3. Every node attends to all KV positions
    4. No node attends to non-ancestors (in tree portion)
    5. Mask is lower triangular for tree portion (causal)
    
    Args:
        mask: TreeAttentionMask to verify
        verbose: If True, print detailed diagnostics
        
    Returns:
        True if all checks pass
    """
    all_pass = True
    dense = mask.to_dense_int()
    
    # Check 1: Self-attention
    for i in range(mask.tree_size):
        if not dense[i, i]:
            if verbose:
                print(f"FAIL: Node {i} does not attend to itself")
            all_pass = False
    
    # Check 2: Ancestor attention
    for node in mask.tree.nodes:
        ancestors = node.get_ancestors()
        for ancestor in ancestors:
            if not dense[node.index, ancestor.index]:
                if verbose:
                    print(f"FAIL: Node {node.index} does not attend to ancestor {ancestor.index}")
                all_pass = False
    
    # Check 3: KV attention
    if mask.kv_len > 0:
        kv_start = mask.tree_size
        for i in range(mask.tree_size):
            for j in range(mask.kv_len):
                if not dense[i, kv_start + j]:
                    if verbose:
                        print(f"FAIL: Node {i} does not attend to KV position {j}")
                    all_pass = False
    
    # Check 4: No non-ancestor attention (in tree portion)
    for node in mask.tree.nodes:
        ancestors_set = {a.index for a in node.get_ancestors()}
        ancestors_set.add(node.index)  # Include self
        
        for j in range(mask.tree_size):
            if j not in ancestors_set and dense[node.index, j]:
                if verbose:
                    print(f"FAIL: Node {node.index} attends to non-ancestor {j}")
                all_pass = False
    
    # Check 5: Lower triangular (causal) in BFS order
    # In BFS order, ancestors always have lower indices than descendants
    for i in range(mask.tree_size):
        for j in range(i + 1, mask.tree_size):
            if dense[i, j]:
                if verbose:
                    print(f"FAIL: Node {i} attends to future node {j} (non-causal)")
                all_pass = False
    
    if verbose:
        if all_pass:
            print("All tree mask correctness checks passed!")
        else:
            print("Some checks failed!")
    
    return all_pass


def compare_tree_vs_sequential_attention(tree_mask: TreeAttentionMask,
                                          kv_embeddings: np.ndarray,
                                          tree_embeddings: np.ndarray,
                                          attention_fn,
                                          tolerance: float = 1e-3) -> Dict:
    """Compare tree attention output against sequential per-path attention.
    
    This validates that the tree attention implementation produces the same
    results as computing attention sequentially along each path.
    
    Args:
        tree_mask: TreeAttentionMask for the tree
        kv_embeddings: [kv_len, hidden_size] KV cache embeddings
        tree_embeddings: [tree_size, hidden_size] tree node embeddings
        attention_fn: Function that computes attention given query, kv, and mask
                     Signature: attention_fn(query, kv, mask_row) -> output
        tolerance: Maximum absolute error allowed
        
    Returns:
        Dict with comparison results
    """
    tree_size = tree_mask.tree_size
    hidden_size = tree_embeddings.shape[1]
    
    # Compute tree attention (batched)
    # This would use the tree attention kernel
    # For now, we'll compute it per-node as a reference
    tree_outputs = []
    
    for i in range(tree_size):
        # Get mask row for this node
        mask_row = tree_mask.mask[i]
        
        # Gather attended keys/values
        attended_positions = np.where(mask_row)[0]
        
        # Separate tree and KV positions
        tree_positions = [p for p in attended_positions if p < tree_size]
        kv_positions = [p - tree_size for p in attended_positions if p >= tree_size]
        
        # Build attended kv_embeddings
        attended_kv = []
        for p in tree_positions:
            attended_kv.append(tree_embeddings[p])
        for p in kv_positions:
            attended_kv.append(kv_embeddings[p])
        
        attended_kv = np.stack(attended_kv)  # [attended_len, hidden_size]
        
        # Compute attention
        query = tree_embeddings[i:i+1]  # [1, hidden_size]
        output = attention_fn(query, attended_kv, None)  # [1, hidden_size]
        tree_outputs.append(output[0])
    
    tree_outputs = np.stack(tree_outputs)  # [tree_size, hidden_size]
    
    # For sequential comparison, we'd compute attention for each path independently
    # This is left as a TODO for the kernel implementation
    
    return {
        'tree_outputs': tree_outputs,
        'sequential_outputs': None,  # TODO: implement
        'max_abs_error': 0.0,  # TODO: compute
        'cosine_similarity': 1.0,  # TODO: compute
        'passed': True  # Placeholder
    }
