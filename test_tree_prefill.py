#!/usr/bin/env python3
"""Minimal test to verify tree speculative uses prefill_step."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.tree_speculative import TreeSpeculativeDecoder
import inspect

print('Import successful - TreeSpeculativeDecoder module loads correctly')
print('Checking if _verify_tree method uses prefill_step...')
source = inspect.getsource(TreeSpeculativeDecoder._verify_tree)
if 'prefill_step' in source:
    print('SUCCESS: _verify_tree uses prefill_step as expected')
else:
    print('ERROR: _verify_tree does NOT use prefill_step')

print('\nFirst 600 chars of source:')
print(source[:600])
