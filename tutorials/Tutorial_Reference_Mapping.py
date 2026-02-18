# %%
import argparse
import os
import sys
from pathlib import Path
import warnings

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import torch first to avoid DLL issues on Windows
import torch

import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import matplotlib.pyplot as plt
from scgpt.preprocess import Preprocessor
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
import scgpt as scg

# extra dependency for similarity search
try:
    import faiss

    faiss_imported = True
except ImportError:
    faiss_imported = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")

warnings.filterwarnings("ignore", category=ResourceWarning)

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

## Referrence mapping using a customized reference dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='ms',help='dataset name (ms/zheng68k/COVID/NSCLC/brain_cellxgene).')
parser.add_argument("--data_path", type=str, default='data', help='Path to data directory.')
parser.add_argument("--model_path", type=str, default='scGPT_human', help='Path to model directory.')
parser.add_argument("--load_model", type=str, default=None, help='Path to pretrained model (alternative to --model_path).')
parser.add_argument("--subset_train", type=int, default=None, help='Subset training data to N cells (None=all, e.g. 500 for quick test)')
parser.add_argument("--subset_val", type=int, default=None, help='Subset validation data to N cells (None=all, for compatibility)')
parser.add_argument("--subset_test", type=int, default=None, help='Subset test data to N cells (None=all)')
parser.add_argument("--batch_size", type=int, default=16, help='Batch size for embedding (increase for speed).')
parser.add_argument("--max_length", type=int, default=256, help='Max sequence length for embedding (reduce for speed).')
parser.add_argument("--k", type=int, default=10, help='Number of nearest neighbors for similarity search.')
parser.add_argument("--num_workers", type=int, default=4, help='Number of workers for dataloader.')
parser.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility.')
parser.add_argument("--save_dir", type=str, default=None, help='Directory to save results.')
parser.add_argument("--verbose", type=bool, default=True, help='Verbose output.')
parser.add_argument("--use_gpu", type=bool, default=True, help='Use GPU if available (faster).')
parser.add_argument("--skip_preprocessing", type=bool, default=False, help='Skip preprocessing if already done.')
parser.add_argument("--cache_embeddings", type=bool, default=False, help='Cache embeddings for reuse.')
args = parser.parse_args()
# %%
import time
start_time = time.time()

data_name = args.data_name
data_path = args.data_path
model_path = args.load_model if args.load_model else args.model_path
subset_train = args.subset_train
subset_val = args.subset_val
subset_test = args.subset_test
batch_size = args.batch_size
max_length = args.max_length
k_neighbors = args.k
seed = args.seed
verbose = args.verbose
use_gpu = args.use_gpu and torch.cuda.is_available()
skip_preprocessing = args.skip_preprocessing
cache_embeddings = args.cache_embeddings

# Set seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

if use_gpu:
    torch.cuda.manual_seed(seed)
    if verbose:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    if verbose:
        print("ℹ Using CPU (slower). Add --use_gpu True to enable GPU.")

if verbose:
    print("\n" + "="*70)
    print(f"CONFIG: batch_size={batch_size}, max_length={max_length}, k={k_neighbors}")
    print(f"SUBSET:  train={subset_train}, test={subset_test}")
    print("="*70 + "\n")

if os.path.isabs(model_path):
    model_dir = Path(model_path)
else:
    model_dir = project_root / model_path

if not (model_dir / "args.json").exists():
    print(f"Error: 'args.json' not found in {model_dir}.")
    print("The scGPT model requires 'args.json' for configuration.")
    if (model_dir / "config.json").exists():
        print("Found 'config.json'. Please rename it to 'args.json'.")
    sys.exit(1)

if os.path.isabs(data_path):
    data_root = Path(data_path)
else:
    data_root = project_root / data_path

data_dir = data_root / data_name if (data_root / data_name).exists() else data_root

# Handle different file naming conventions
if data_name == 'brain_cellxgene':
    train_path = data_dir / "cxg_train.h5ad"
    test_path = data_dir / "cxg_test.h5ad"
else:
    train_path = data_dir / f"{data_name}_train.h5ad"
    test_path = data_dir / f"{data_name}_test.h5ad"

adata = sc.read_h5ad(train_path)

# Subset training data if requested
if subset_train is not None:
    if verbose:
        print(f"Subsetting training data to {subset_train} cells...")
    adata = adata[:subset_train].copy()
    if verbose:
        print(f"✓ Training data shape: {adata.shape}")

if data_name == 'ms':
    data_is_raw = False
    celltype_key = 'celltype'
elif data_name == 'zheng68k':
    data_is_raw = False
    celltype_key = 'celltype'
elif data_name == 'brain_cellxgene':
    data_is_raw = False
    celltype_key = 'cell_type'
elif data_name == 'COVID':
    data_is_raw = True
    celltype_key = 'cell_type'
elif data_name == 'NSCLC':
    data_is_raw = True
    celltype_key = 'cell_type'

gene_col = "index"

# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=51,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
if data_is_raw:
    preprocessor(adata, batch_key=None)
    adata.X = adata.layers['X_log1p']
else:
    preprocessor(adata, batch_key=None)
    adata.X = adata.layers['X_normed']

if verbose:
    print(f"\n[1/4] Embedding training data ({adata.shape[0]} cells)...")
emb_start = time.time()
ref_embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    cell_type_key=celltype_key,
    max_length=max_length,
    gene_col=gene_col,
    batch_size=batch_size,
    return_new_adata=True,
)
if verbose:
    print(f"✓ Training embedding done in {time.time() - emb_start:.1f}s")

test_adata = sc.read_h5ad(test_path)
# Subset test data if requested
if subset_test is not None:
    if verbose:
        print(f"Subsetting test data to {subset_test} cells...")
    test_adata = test_adata[:subset_test].copy()
    if verbose:
        print(f"✓ Test data shape: {test_adata.shape}")
if data_is_raw:
    preprocessor(test_adata, batch_key=None)
    test_adata.X = test_adata.layers['X_log1p']
else:
    preprocessor(test_adata, batch_key=None)
    test_adata.X = test_adata.layers['X_normed']

if verbose:
    print(f"\n[2/4] Embedding test data ({test_adata.shape[0]} cells)...")
emb_start = time.time()
test_embed_adata = scg.tasks.embed_data(
    test_adata,
    model_dir,
    cell_type_key=celltype_key,
    max_length=max_length,
    gene_col=gene_col,
    batch_size=batch_size,
    return_new_adata=True,
)
if verbose:
    print(f"✓ Test embedding done in {time.time() - emb_start:.1f}s")


# Those functions are only used when faiss is not installed
def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims


def get_similar_vectors(vector, ref, top_k=10):
    # sims = cos_sim(vector, ref)
    sims = l2_sim(vector, ref)

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


# %%

ref_cell_embeddings = ref_embed_adata.X
test_emebd = test_embed_adata.X

k = k_neighbors  # number of neighbors from args

if verbose:
    print(f"\n[3/4] Computing similarity search (k={k})...")
sim_start = time.time()

if faiss_imported:
    # Declaring index, using most of the default parameters from
    index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
    index.add(ref_cell_embeddings)

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    distances, labels = index.search(test_emebd, k)
    
    if verbose:
        print(f"✓ Similarity search done in {time.time() - sim_start:.1f}s (using FAISS - GPU accelerated)")
else:
    if verbose:
        print("⚠ FAISS not available, using slower L2 similarity search")

idx_list = [i for i in range(test_emebd.shape[0])]
preds = []
for k in idx_list:
    if faiss_imported:
        idx = labels[k]
    else:
        idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, k)
    pred = ref_embed_adata.obs[celltype_key][idx].mode()[0]
    preds.append(pred[0][0])

if verbose:
    print(f"\n[4/4] Computing metrics...")

gt = test_adata.obs[celltype_key].to_numpy()
train_label_dict, train_label = np.unique(np.array(adata.obs[celltype_key]), return_inverse=True)
truths = adata.obs[celltype_key].tolist()

weighted_f1 = f1_score(gt, preds, average='weighted')
balanced_accuracy = balanced_accuracy_score(gt, preds)
precision = precision_score(gt, preds, average="weighted")
recall = recall_score(gt, preds, average="weighted")

elapsed_time = time.time() - start_time

if verbose:
    print("\n" + "="*70)
    print(classification_report(gt, preds, digits=4))
    print("="*70)
    print(f'F1 Score: {weighted_f1:.6f} | Acc: {balanced_accuracy * 100:.4f}% | precision: {precision:.6f} | recall: {recall:.6f}')
    print(f"✓ Total time: {elapsed_time:.1f}s")
    print("="*70)

# Save results if save_dir is provided
if args.save_dir:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'predictions': preds,
        'ground_truth': gt.tolist(),
        'f1_score': float(weighted_f1),
        'accuracy': float(balanced_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'elapsed_time_seconds': float(elapsed_time),
        'config': {
            'data_name': data_name,
            'subset_train': subset_train,
            'subset_test': subset_test,
            'batch_size': batch_size,
            'max_length': max_length,
            'k_neighbors': k_neighbors,
            'seed': seed,
            'use_gpu': use_gpu,
        }
    }
    
    import json
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n✓ Results saved to {save_dir / 'results.json'}")
