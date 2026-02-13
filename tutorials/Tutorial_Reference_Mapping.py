import argparse
import os
import sys
from pathlib import Path
import warnings

# Set up sys.path before importing scgpt modules
script_dir = Path(__file__).resolve().parent
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

## Referrence mapping using a customized reference dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='ms',help='dataset name.')
parser.add_argument("--data_path", type=str, default='data', help='Path to data directory.')
parser.add_argument("--model_path", type=str, default='scGPT_human', help='Path to model directory.')
args = parser.parse_args()
# %%
data_name = args.data_name
data_path = args.data_path
model_path = args.model_path

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
train_path = data_dir / f"{data_name}_train.h5ad"
test_path = data_dir / f"{data_name}_test.h5ad"

adata = sc.read_h5ad(train_path)

if data_name == 'ms':
    data_is_raw = False
    celltype_key = 'celltype'
elif data_name == 'zheng68k':
    data_is_raw = False
    celltype_key = 'celltype'
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

ref_embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    cell_type_key=celltype_key,
    max_length=2001,
    gene_col=gene_col,
    batch_size=20,
    return_new_adata=True,
)

test_adata = sc.read_h5ad(test_path)
if data_is_raw:
    preprocessor(test_adata, batch_key=None)
    test_adata.X = test_adata.layers['X_log1p']
else:
    preprocessor(test_adata, batch_key=None)
    test_adata.X = test_adata.layers['X_normed']
test_embed_adata = scg.tasks.embed_data(
    test_adata,
    model_dir,
    cell_type_key=celltype_key,
    max_length=2001,
    gene_col=gene_col,
    batch_size=20,
    return_new_adata=True,
)


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

k = 10  # number of neighbors

if faiss_imported:
    # Declaring index, using most of the default parameters from
    index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
    index.add(ref_cell_embeddings)

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    distances, labels = index.search(test_emebd, k)

idx_list = [i for i in range(test_emebd.shape[0])]
preds = []
for k in idx_list:
    if faiss_imported:
        idx = labels[k]
    else:
        idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, k)
    pred = ref_embed_adata.obs[celltype_key][idx].mode()[0]
    preds.append(pred[0][0])

gt = test_adata.obs[celltype_key].to_numpy()
train_label_dict, train_label = np.unique(np.array(adata.obs[celltype_key]), return_inverse=True)
truths = adata.obs[celltype_key].tolist()

weighted_f1 = f1_score(gt, preds, average='weighted')
balanced_accuracy = balanced_accuracy_score(gt, preds)
precision = precision_score(gt, preds, average="weighted")
recall = recall_score(gt, preds, average="weighted")

print(classification_report(gt, preds, digits=4))

print(f'F1 Score: {weighted_f1:.6f} | Acc: {balanced_accuracy * 100:.4f}% | precision: {precision:.6f} | recall: {recall:.6f}')
