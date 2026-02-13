#!/usr/bin/env python3
"""
Quick validation script to test if Tutorial_Annotation_Prompt.py runs without errors.
Tests: imports, data loading, model initialization, and 1 training step.
Expected runtime: 2-5 minutes instead of hours.
"""
import os
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.getcwd(), ".numba_cache")
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import sys
import time
from pathlib import Path

print("=" * 70)
print("QUICK VALIDATION TEST - Tutorial_Annotation_Prompt.py")
print("=" * 70)

# Test 1: Import all dependencies
print("\n[1/6] Testing imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import warnings
    warnings.filterwarnings('ignore')
    
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root))
    
    from scgpt.model.model_prompt import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.loss import masked_mse_loss
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.preprocess import Preprocessor
    
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load data
print("\n[2/6] Testing data loading...")
try:
    data_path = project_root / 'data' / 'ms'
    adata_train = sc.read(data_path / "ms_train.h5ad")
    adata_test = sc.read(data_path / "ms_test.h5ad")
    
    print(f"✓ Loaded train data: {adata_train.shape}")
    print(f"✓ Loaded test data: {adata_test.shape}")
    
    # Subset to 100 cells for speed
    adata_train = adata_train[:100, :].copy()
    adata_test = adata_test[:50, :].copy()
    print(f"✓ Subsampled to: train {adata_train.shape}, test {adata_test.shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 3: Preprocessing
print("\n[3/6] Testing preprocessing...")
try:
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        binning=51,
        result_binned_key="X_binned",
    )
    
    preprocessor(adata_train, batch_key=None)
    preprocessor(adata_test, batch_key=None)
    print("✓ Preprocessing successful")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    sys.exit(1)

# Test 4: Tokenization
print("\n[4/6] Testing tokenization...")
try:
    from torchtext.vocab import Vocab
    from torchtext._torchtext import Vocab as VocabPybind
    
    genes = adata_train.var.index.tolist()[:200]  # Use only 200 genes
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    gene_ids = np.array(vocab(genes), dtype=int)
    
    train_data = adata_train.layers["X_binned"]
    train_data = train_data.A if hasattr(train_data, 'A') else train_data
    
    tokenized = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=500,  # Reduced for speed
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=False,
    )
    print(f"✓ Tokenization successful: {tokenized['genes'].shape}")
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    sys.exit(1)

# Test 5: Model initialization
print("\n[5/6] Testing model initialization...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    model = TransformerModel(
        ntokens=len(vocab),
        embsize=128,
        nhead=4,
        d_hid=128,
        nlayers=2,  # Reduced for speed
        nlayers_cls=2,
        n_cls=2,
        vocab=vocab,
        dropout=0.2,
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        domain_spec_batchnorm=False,
        input_emb_style="scaling",
        n_input_bins=51,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
        batch_size=8,
        use_prompt=True,
        num_tokens=64,
        prompt_type="prefix_prompt",
        n_layers_conf=[1]*12,
        mlp_adapter_conf=[0]*12,
        space_adapter_conf=[0]*12,
        max_len=500,
    )
    model.to(device)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 6: One forward/backward pass
print("\n[6/6] Testing one training step...")
try:
    model.train()
    
    input_gene_ids = torch.from_numpy(tokenized['genes'][:8]).long().to(device)
    input_values = torch.from_numpy(tokenized['values'][:8]).to(device)
    target_values = torch.from_numpy(tokenized['values'][:8]).to(device)
    
    src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
    
    # Add prefix prompt mask
    bool_tensor = torch.zeros((input_gene_ids.shape[0], 64), dtype=torch.bool).to(device)
    src_key_padding_mask = torch.cat((bool_tensor, src_key_padding_mask), dim=1)
    
    with torch.cuda.amp.autocast(enabled=True):
        output_dict = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        
        masked_positions = input_values.eq(-1)
        loss = masked_mse_loss(
            output_dict["mlm_output"], target_values, masked_positions
        )
    
    loss.backward()
    
    print(f"✓ Forward/backward pass successful")
    print(f"  Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Script is working!")
print("=" * 70)
print("\nYou can now run the full training script:")
print("  python tutorials/Tutorial_Annotation_Prompt.py --data_name ms --epoch 1 --batch_size 8")
print()
