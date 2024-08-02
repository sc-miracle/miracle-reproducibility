
MIRACLE is a continual integration method for single-cell mosaic data (RNA, ADT, and ATAC are currently supported). In this repository, we mainly repreduce the results in the paper. For tutorials, please watch our another repository.

# Preparation

Use the following command to install the required packages.

```bash
conda install -i requirements.txt
```

**Main packages:**

(Python)

Python==3.11

PyTorch==2.1.2

scikit-learn==1.3.2

scanpy==1.9.6

scib==1.1.4

(R)

seurat== 4.1.0

signac ==1.6.0

# Horizontal integration (DCM_HCM)

## Processing data

For quality control, please see `reproducibility-code/result2-DCM_HCM/proprocess_dcm_hcm.ipynb`.

Generating 42 bathes of data (in this experiement, all batches are considered with the same features):

```bash
Rscript preprocess/combine_subsets.R --task dcm_hcm
python preprocess/split_mat.py --task dcm_hcm
for i in {0..41}
do
mkdir ./data/processed/dcm_hcm_$[i+1]
mkdir ./data/processed/dcm_hcm_$[i+1]/subset_0
ln -sr ./data/processed/dcm_hcm/subset_$i/* ./data/processed/dcm_hcm_$[i+1]/subset_0/
ln -sr ./data/processed/dcm_hcm/feat ./data/processed/dcm_hcm_$[i+1]/
done
```

## Training MIRACLE

```bash
python train.py --cuda 0 --task dcm_hcm --exp_prefix continual_ --max_size 50000 --actions train predict_all_latent
```

## Evaluation

```bash
python eval/benchmark_batch_bio.py --task dcm_hcm --experiment continual_41
```

## Visualization

It takes a long time to compute the UMAP for the variable u. Therefore, we suggest you to visualize the UMAP only for the variable c by setting 'use_u' to 0.

```bash
Rscript comparison/midas_embed.r --task dcm_hcm --experiment continual_41 --use_u 0
```

The obtained seurat object contains the UMAP embeddings and you can visualize them referring to the notebook `~/MIRACLE-reproducibility/reproducibility-code/result2-DCM_HCM/visualize_results.ipynb`

## Comparison of subsampling methods

see:

```
reproducibility-code/result2-DCM_HCM/compare_subsampling.ipynb
```

# Rectangular integration (WNN)

## Generating data

Generating 8 bathes of data:

```bash
Rscript preprocess/combine_subsets.R --task p1_0 && py preprocess/split_mat.py --task p1_0 & 
Rscript preprocess/combine_subsets.R --task p2_0 && py preprocess/split_mat.py --task p2_0 & 
Rscript preprocess/combine_subsets.R --task p3_0 && py preprocess/split_mat.py --task p3_0 & 
Rscript preprocess/combine_subsets.R --task p4_0 && py preprocess/split_mat.py --task p4_0 & 
Rscript preprocess/combine_subsets.R --task p5_0 && py preprocess/split_mat.py --task p5_0 & 
Rscript preprocess/combine_subsets.R --task p6_0 && py preprocess/split_mat.py --task p6_0 & 
Rscript preprocess/combine_subsets.R --task p7_0 && py preprocess/split_mat.py --task p7_0 & 
Rscript preprocess/combine_subsets.R --task p8_0 && py preprocess/split_mat.py --task p8_0
```

## Training MIRACLE

```bash
python train.py --cuda 0 --task wnn --exp_prefix continual_ --actions train predict_all_latent
```

## Evaluation

```bash
python eval/benchmark_batch_bio.py --task wnn --experiment continual_7
```

## Visualization

```bash
Rscript comparison/midas_embed.r --task wnn --experiment continual_7
```

# Mosaic integration (TEADOG)

## Generating training data

Generating 8 different teadog mosaic datasets:

```bash
Rscript preprocess/combine_subsets.R --task teadog_label_mask && py preprocess/split_mat.py --task teadog_label_mask # reference

Rscript preprocess/combine_subsets.R --task lll_ctrl && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task lll_ctrl && py preprocess/split_mat.py --task lll_ctrl &
Rscript preprocess/combine_subsets.R --task lll_stim && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task lll_stim && py preprocess/split_mat.py --task lll_stim &
Rscript preprocess/combine_subsets.R --task dig_ctrl && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task dig_ctrl && py preprocess/split_mat.py --task dig_ctrl &
Rscript preprocess/combine_subsets.R --task dig_stim && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task dig_stim && py preprocess/split_mat.py --task dig_stim &
Rscript preprocess/combine_subsets.R --task W3 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W3 && py preprocess/split_mat.py --task W3 &
Rscript preprocess/combine_subsets.R --task W4 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W4 && py preprocess/split_mat.py --task W4 &
Rscript preprocess/combine_subsets.R --task W5 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W5 && py preprocess/split_mat.py --task W5 &
Rscript preprocess/combine_subsets.R --task W6 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W6 && py preprocess/split_mat.py --task W6
```

## Training MIRACLE

```bash
python train.py --cuda 0 --task teadog --exp_prefix continual_ --actions train predict_all_latent
```

## Evaluation

```bash
python eval/benchmark_batch_bio.py --task teadog --experiment continual_7
```

## Visualization

```bash
Rscript comparison/midas_embed.r --task teadog --experiment continual_7
```

# Comparison integration methods

pleaser refer to the **reproducibility-code** directory

# Continual atlas construction

## Integrate atlas with MIRACLE-offline

```bash
Rscript preprocess/combine_subsets.R --task atlas_new_no_neap
py preprocess/split_mat.py --task atlas_new_no_neap
CUDA_VISIBLE_DEVICES=0 python run.py --task atlas_new_no_neap --experiment offline --actions train predict_all_latent --use_shm 1

Rscript preprocess/combine_subsets.R --task atlas_new
py preprocess/split_mat.py --task atlas_new
CUDA_VISIBLE_DEVICES=1 python run.py --task atlas_new --experiment offline --actions train predict_all_latent --use_shm 1

Rscript preprocess/combine_subsets.R --task atlas_tissues
py preprocess/split_mat.py --task atlas_tissues
CUDA_VISIBLE_DEVICES=2 python run.py --task atlas_tissues --experiment offline --actions train predict_all_latent --use_shm 1
```

## Continually integrate PBMC query datasets into reference dataset

```bash
Rscript preprocess/combine_subsets.R --task query_neat
Rscript preprocess/combine_subsets.R --task asap
Rscript preprocess/combine_subsets.R --task asap_cite
Rscript preprocess/combine_unseen.R --reference atlas_new_no_neap --task query_neat
Rscript preprocess/combine_unseen.R --reference atlas_new_no_neap --task asap
py preprocess/split_mat.py --task query_neat
py preprocess/split_mat.py --task asap
py preprocess/split_mat.py --task asap_cite
python train.py --cuda 3 --task new_query_cl --actions train predict_subsample subsample predict_all --denovo 0
```

## Continually integrate cross-tissue query datasets into reference dataset

```bash
Rscript preprocess/combine_subsets.R --task tonsil
Rscript preprocess/combine_subsets.R --task bone_marrow_02
Rscript preprocess/combine_subsets.R --task spleen
Rscript preprocess/combine_unseen.R --reference atlas_new --task bone_marrow_02
py preprocess/split_mat.py --task tonsil
py preprocess/split_mat.py --task bone_marrow_02
py preprocess/split_mat.py --task spleen
python train.py --cuda 4 --task atlas_tissues_cl --actions train predict_subsample subsample predict_all --denovo 0
```

# Accurate label transfer

## Label transfer of PBMC query datasets

```bash
python train.py --cuda 0 --task single_query_neat_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 1 --task single_asap_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 2 --task single_asap_cite_cl --actions train predict_subsample subsample predict_all --denovo 0
```

## Label transfer of cross-tissue query datasets

```bash
python train.py --cuda 3 --task single_tonsil_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 4 --task single_bm_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 5 --task single_spleen_cl --actions train predict_subsample subsample predict_all --denovo 0
```
