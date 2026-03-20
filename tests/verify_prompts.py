#!/usr/bin/env python3
"""
tests/verify_prompts.py — Verification Script for Prompt Module.

Verifies that all requirements for m2-prepare-real-text-prompts are met:
  1. Test prompts created for all 4 domains
  2. Train/test split implemented correctly (60%/40%)
  3. Prompts cover various patterns and structures
  4. Tokenizer integration ready

USAGE:
    python3 tests/verify_prompts.py
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_prompt_module():
    """Verify prompt module exists and loads correctly."""
    print("=" * 72)
    print("  1. Prompt Module Verification")
    print("=" * 72)
    
    try:
        from src.inference.prompts import PromptDataset, PromptCategory, TrainTestSplitter, TokenizerAdapter
        print("  ✓ Prompt module imports successfully")
        
        # Test dataset creation
        dataset = PromptDataset()
        print(f"  ✓ PromptDataset created: {dataset}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import prompt module: {e}")
        return False


def verify_four_domains():
    """Verify test prompts exist for all 4 domains."""
    print("\n" + "=" * 72)
    print("  2. Four Domain Categories Verification")
    print("=" * 72)
    
    from src.inference.prompts import PromptDataset, PromptCategory
    
    dataset = PromptDataset()
    required_categories = {
        'code': 'Code completion (Python functions, classes)',
        'json': 'JSON completion (nested structures)',
        'conversational': 'Conversational (dialogue)',
        'repetitive': 'Repetitive text (high n-gram match rate)'
    }
    
    all_present = True
    for cat_name, description in required_categories.items():
        category = PromptCategory(cat_name)
        prompts = dataset.get_prompts(category)
        
        if len(prompts) > 0:
            print(f"  ✓ {cat_name.upper()}: {len(prompts)} prompts")
            print(f"      Description: {description}")
            
            # Show first prompt as example
            if prompts:
                sample = prompts[0].text[:60].replace('\n', '\\n')
                print(f"      Sample: {sample}...")
        else:
            print(f"  ✗ {cat_name.upper()}: No prompts found")
            all_present = False
    
    return all_present


def verify_train_test_split():
    """Verify train/test split methodology (60%/40%)."""
    print("\n" + "=" * 72)
    print("  3. Train/Test Split Verification (60%/40%)")
    print("=" * 72)
    
    from src.inference.prompts import PromptDataset, TrainTestSplitter
    
    dataset = PromptDataset()
    splitter = TrainTestSplitter(train_ratio=0.6)
    
    # Test on each category
    all_valid = True
    for category in dataset.get_by_category_dict().keys():
        prompts = dataset.get_prompts(getattr(__import__('src.inference.prompts', fromlist=['PromptCategory']), 'PromptCategory')(category))
        if prompts:
            sample_text = prompts[0].text
            
            # Test split
            train_text, test_text = splitter.split_text(sample_text)
            train_ratio_actual = len(train_text) / len(sample_text)
            
            # Verify ratio is approximately 60/40 (within 10% tolerance)
            ratio_valid = 0.5 <= train_ratio_actual <= 0.7
            
            status = "✓" if ratio_valid else "✗"
            print(f"  {status} {category}: train={len(train_text)} ({train_ratio_actual:.0%}), test={len(test_text)} ({1-train_ratio_actual:.0%})")
            
            if not ratio_valid:
                all_valid = False
    
    # Test token split
    print("\n  Token-level split test:")
    sample_tokens = [ord(c) % 256 for c in dataset.get_prompts(__import__('src.inference.prompts', fromlist=['PromptCategory']).PromptCategory.CODE)[0].text]
    train_tokens, test_tokens = splitter.split_tokens(sample_tokens)
    token_ratio = len(train_tokens) / len(sample_tokens)
    
    token_valid = 0.5 <= token_ratio <= 0.7
    status = "✓" if token_valid else "✗"
    print(f"  {status} Token split: train={len(train_tokens)} ({token_ratio:.0%}), test={len(test_tokens)} ({1-token_ratio:.0%})")
    
    return all_valid and token_valid


def verify_prompt_variety():
    """Verify prompts cover various patterns and structures."""
    print("\n" + "=" * 72)
    print("  4. Prompt Variety Verification")
    print("=" * 72)
    
    from src.inference.prompts import PromptDataset, PromptCategory
    
    dataset = PromptDataset()
    
    # Check each category has multiple prompts with different patterns
    all_valid = True
    for category in PromptCategory:
        prompts = dataset.get_prompts(category)
        
        if len(prompts) < 2:
            print(f"  ✗ {category.value}: Only {len(prompts)} prompt(s), need at least 2")
            all_valid = False
            continue
        
        # Check for variety in metadata
        patterns = set(p.metadata.get('pattern', 'unknown') for p in prompts)
        descriptions = set(p.description for p in prompts)
        
        if len(patterns) > 1 or len(descriptions) > 1:
            print(f"  ✓ {category.value}: {len(prompts)} prompts with {len(patterns)} different pattern(s)")
            for p in prompts:
                print(f"      - {p.description} (pattern: {p.metadata.get('pattern', 'N/A')})")
        else:
            print(f"  ⚠ {category.value}: All prompts have same pattern ({patterns})")
    
    return all_valid


def verify_tokenizer_integration():
    """Verify tokenizer integration is ready."""
    print("\n" + "=" * 72)
    print("  5. Tokenizer Integration Verification")
    print("=" * 72)
    
    from src.inference.prompts import TokenizerAdapter
    
    try:
        # Test character-level tokenizer
        tokenizer = TokenizerAdapter.create_char_level()
        sample_text = "Test tokenizer integration with Python code."
        
        # Encode
        tokens = tokenizer.encode(sample_text)
        print(f"  ✓ Character-level tokenizer created")
        print(f"      Input: {sample_text}")
        print(f"      Encoded: {len(tokens)} tokens")
        
        # Decode
        decoded = tokenizer.decode(tokens)
        roundtrip_ok = decoded == sample_text
        status = "✓" if roundtrip_ok else "✗"
        print(f"  {status} Round-trip: {'successful' if roundtrip_ok else 'failed'}")
        
        if not roundtrip_ok:
            print(f"      Original: {sample_text}")
            print(f"      Decoded: {decoded}")
            return False
        
        # Test batch encoding
        batch_texts = ["First prompt", "Second prompt", "Third prompt"]
        batch_tokens = tokenizer.encode_batch(batch_texts)
        print(f"  ✓ Batch encoding: {len(batch_tokens)} sequences")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Tokenizer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_json_file():
    """Verify JSON file exists and loads correctly."""
    print("\n" + "=" * 72)
    print("  6. JSON File Verification")
    print("=" * 72)
    
    json_path = Path(__file__).parent.parent / "data" / "test_prompts.json"
    
    if not json_path.exists():
        print(f"  ✗ JSON file not found: {json_path}")
        return False
    
    print(f"  ✓ JSON file exists: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Verify structure
        if 'categories' not in data:
            print(f"  ✗ JSON missing 'categories' key")
            return False
        
        categories = data['categories']
        print(f"  ✓ JSON structure valid with {len(categories)} categories")
        
        for cat_name, prompts in categories.items():
            print(f"      - {cat_name}: {len(prompts)} prompts")
        
        # Verify we can load it back
        from src.inference.prompts import PromptDataset
        dataset = PromptDataset.from_file(str(json_path))
        print(f"  ✓ JSON loads successfully: {dataset}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ JSON validation failed: {e}")
        return False


def verify_val_m6_integration():
    """Verify val_m6_speculative.py integrates with prompt module."""
    print("\n" + "=" * 72)
    print("  7. val_m6_speculative.py Integration Verification")
    print("=" * 72)
    
    val_path = Path(__file__).parent / "val_m6_speculative.py"
    
    if not val_path.exists():
        print(f"  ✗ val_m6_speculative.py not found")
        return False
    
    print(f"  ✓ val_m6_speculative.py exists")
    
    # Check if it imports prompt module
    with open(val_path, 'r') as f:
        content = f.read()
    
    if 'from src.inference.prompts import' in content:
        print(f"  ✓ Imports prompt module")
    else:
        print(f"  ⚠ Does not import prompt module (using fallback)")
    
    if 'PromptDataset' in content or 'TEST_PROMPTS' in content:
        print(f"  ✓ Uses prompt dataset or TEST_PROMPTS")
    else:
        print(f"  ✗ Missing prompt integration")
        return False
    
    return True


def main():
    """Run all verification checks."""
    print_header()
    
    results = []
    
    # Run all verifications
    results.append(("Prompt Module", verify_prompt_module()))
    results.append(("Four Domains", verify_four_domains()))
    results.append(("Train/Test Split", verify_train_test_split()))
    results.append(("Prompt Variety", verify_prompt_variety()))
    results.append(("Tokenizer Integration", verify_tokenizer_integration()))
    results.append(("JSON File", verify_json_file()))
    results.append(("val_m6 Integration", verify_val_m6_integration()))
    
    # Summary
    print("\n" + "=" * 72)
    print("  VERIFICATION SUMMARY")
    print("=" * 72)
    
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    
    print(f"\n  Total checks: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Pass rate: {passed/total:.0%}")
    
    print("\n  Results:")
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"    {status} {name}")
    
    # Final verdict
    print("\n" + "=" * 72)
    if all(ok for _, ok in results):
        print("  ✓ ALL REQUIREMENTS MET - m2-prepare-real-text-prompts COMPLETE")
        print("=" * 72)
        return 0
    else:
        print("  ✗ SOME REQUIREMENTS NOT MET - REVIEW FAILURES ABOVE")
        print("=" * 72)
        return 1


def print_header():
    print()
    print("=" * 72)
    print("  M2-PREPARE-REAL-TEXT-PROMPTS - VERIFICATION")
    print("=" * 72)
    print()
    print("  Feature: Prepare real text prompts for speculative decoding validation")
    print("  Requirements:")
    print("    1. Test prompts created for all 4 domains")
    print("    2. Train/test split implemented (60%/40%)")
    print("    3. Prompts cover various patterns and structures")
    print("    4. Tokenizer integration ready")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
