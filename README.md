# Multi-fidelity GNNs for drug discovery and quantum mechanics
🎉
Now published in [Nature Communications](https://www.nature.com/articles/s41467-024-45566-8)!
##
![](main-figure.png)
## Abstract
We investigate the potential of graph neural networks (GNNs) for transfer learning and improving molecular property prediction on sparse and expensive to acquire high-fidelity data by leveraging low-fidelity measurements as an inexpensive proxy for a targeted property of interest. This problem arises in discovery processes that rely on funnels or screening cascades for trading off the overall costs against throughput and accuracy. Typically, individual stages in these processes are loosely connected and each one generates data at different scale and fidelity. We consider this setup holistically and demonstrate empirically that existing transfer learning techniques for GNNs are generally unable to harness the information from multi-fidelity cascades. We propose several effective transfer learning approaches based on GNNs including: i) feature augmentation using the outputs of supervised models trained on low-fidelity data (e.g. latent space embeddings or predictions generated using supervised variational graph autoencoders), and ii) transfer learning with adaptive readouts, where pre-training and fine-tuning the readout layer plays a key role in improving the performance on sparse tasks. We study both a transductive setting, where each high-fidelity data point has an associated low-fidelity measurement, as well as the more challenging inductive setup where low-fidelity measurements are not available and need to be extrapolated, which is common in drug discovery. Our empirical evaluation involves a novel drug discovery collection of more than 28 million unique experimental protein-ligand interactions across 37 different targets and 12 quantum properties from the dataset QMugs, along with baselines ranging from GNNs to random forests and support vector machines. The results indicate that transfer learning leveraging low-fidelity evaluations can improve the accuracy of predictive models on sparse tasks by up to eight times while using an order of magnitude less high-fidelity training data. Moreover, the proposed methods consistently outperform existing transfer learning strategies for graph-structured data on the drug discovery and quantum mechanics datasets.

## General
This repository contains the source code for all the machine learning models presented in the **Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting** paper, as well as instructions on how to run the models and collect metrics.

The public multi-fidelity datasets are now part of a collection named **MF-PCBA** (Multi-Fidelity PubChem BioAssay). The datasets are accessible through a separate repository: [https://github.com/davidbuterez/mf-pcba/](https://github.com/davidbuterez/mf-pcba/)

To minimise possible points of failure, the data acquisition and modelling workflows are split into different steps.

## Reproducibility
All the drug discovery data splits for the non-proprietary data are available in the [MF-PCBA repository](https://github.com/davidbuterez/mf-pcba/), with the QMugs data splits available [here](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery/tree/main/assemble_QMugs/data_indices). The training code sets a global seed using `pytorch_lightning.seed_everything(0)`, which covers PyTorch, NumPy and Python random number generators.

## Example/Demo
An example drug discovery multi-fidelity dataset (AID 1445) is provided in the directory [example_data](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery/tree/main/example_data). The subdirectory `SD` contains a single .csv file (`SD.csv`) corresponding to the low-fidelity data available in this assay. The subdirectory `DR` contains `train.csv`, `validate.csv`, and `test.csv` files corresponding to a train/validation/test split of high-fidelity DR data (the same as the output of the MF-PCBA data acquisition workflow below). The provided high-fidelity example dataset also includes low-fidelity (SD) embeddings for each molecule from separate SD models (based on sum or Set Transformer readouts). Low-fidelity embeddings can be added to the training files following the instructions below.

The following code can be used to train a base (non-augmented) high-fidelity model for the example dataset (replace the input/output directories):
```
python -m multifidelity_gnn.train.train_high_fidelity
--data-path example_data/DR --target-label DR --node-latent-dim 50 --smiles-column neut-smiles
--max-atomic-number 35 --max-num-atoms-in-mol 48 --id-column CID --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --gnn-intermediate-dim 256 --no-use-cuda --task-type regression --batch-size 32
--logging-name AID1445_HF_demo --out-dir out_HF
```

Training a model with experimentally-determined low-fidelity labels:
```
python -m multifidelity_gnn.train.train_high_fidelity
--data-path example_data/DR --target-label DR --node-latent-dim 50 --smiles-column neut-smiles
--max-atomic-number 35 --max-num-atoms-in-mol 48 --id-column CID --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --gnn-intermediate-dim 256 --no-use-cuda --task-type regression --batch-size 32
--logging-name AID1445_HF_LBL_demo --out-dir out_HF --lbl-or-emb lbl --auxiliary-dim 1
--train-auxiliary-data-column-name SD --eval-auxiliary-data-column-name SD

```

Training a model with separately-computed low-fidelity embeddings:
```
python -m multifidelity_gnn.train.train_high_fidelity
--data-path example_data/DR --target-label DR --node-latent-dim 50 --smiles-column neut-smiles
--max-atomic-number 35 --max-num-atoms-in-mol 48 --id-column CID --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --gnn-intermediate-dim 256 --no-use-cuda --task-type regression --batch-size 32
--logging-name AID1445_HF_LBL_demo --out-dir out_HF --lbl-or-emb emb --auxiliary-dim 256
--train-auxiliary-data-column-name STEmbeddings --eval-auxiliary-data-column-name STEmbeddings
```

Note that `--train-auxiliary-data-column-name` and `--eval-auxiliary-data-column-name` can be set to different values to replicate the hybrid experiments described in the paper.

The training script above will produce a `test_metrics.npy` file containing the test set metrics and a `test_output.npy` file containing the predictions for the test set (all generated files are located in the output directory provided to the training script).

The high-fidelity models above are quick enough to require around 1 second per epoch, and less than 1 minute overall training time for the entire training run on a modern laptop. These times are achieved without using a graphics processing unit (CUDA). CUDA is recommended for the larger low-fidelity datasets.

Please check the **Requirements/installation** section below for details regarding the software versions that were tested and compatible hardware.

## QMugs
The SMILES-encoded QMugs dataset can be obtained from the official repository ([ETH Library Collection service](https://www.research-collection.ethz.ch/handle/20.500.11850/482129)). In particular, we use the `summary.csv` file. All the splits described in the paper can be assembled by using the `chembl_id` and `conf_id` indices provided in this repository ([assemble_QMugs/data_indices](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery/tree/main/assemble_QMugs/data_indices)), as exemplified in the [assemble_QMugs/assemble_QMugs.ipynb](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery/tree/main/assemble_QMugs/assemble_QMugs.ipynb) notebook. We have further removed molecules that were more than 5 standard deviations away from the mean (for each of the 12 properties, for both DFT and GFN2-xTB), removing slightly over 1% of the total number of molecules.

All the deep learning training code available in this repository is exemplified on drug discovery data can be used seamlessly with the QMugs data.

## Workflows
### 0. Data acquisition
1. Download one dataset or a selection of datasets using the code and instructions from the MF-PCBA repository. For example, the following command downloads the AID 1445 dataset to a `save_dir` directory:

```
python pubchem_retrieve.py --AID "1445"
--list_of_sd_cols "Primary Inhibition" "Primary Inhibition Rep 2" "Primary Inhibition Rep 3" 
--list_of_dr_cols "IC50" --transform_dr "pXC50" --save_dir <save_dir>
```

2. The step above downloaded and filtered the data corresponding to AID 1445. To obtain train, validation, and test sets, the `split_DR_with_random_seeds.ipynb` notebook available in the MF-PCBA repository can be used. The same 5 random split seeds as used in the paper are provided in the MF-PCBA repository and are used by default. After this step, the high-fidelity DR data is split into train, validation, and test sets 5 different times, with the resulting `.csv` files being saved in different directories:

```
parent_dir/
├── 0/
│   ├── train.csv
│   ├── validate.csv
│   └── test.csv
├── 1/
│   ├── train.csv
│   ├── validate.csv
│   └── test.csv
| ...
└──
...
```

### 1. Training non-augmented models and models augmented with low-fidelity labels
These models do not require a separate low-fidelity modelling phase and can be applied directly to high-fidelity data with train, validation, and test splits. High-fidelity model training is handled by the Python script `train_high_fidelity.py`. An example:

```
python -m multifidelity_gnn.train.train_high_fidelity
--data-path <directory containing train/val/test files>
--out-dir <directory where checkpoints and metrics are saved>
--target-label HF --node-latent-dim 50 --smiles-column neut-smiles
--max-atomic-number 35 --max-num-atoms-in-mol 48 --id-column CID --use-vgae --num-layers 3
--conv GCN --use-batch-norm --gnn-intermediate-dim 256 --logging-name <LOG-NAME>
--task-type regression --no-use-cuda
```

The arguments specify a model with a node dimension of 50, 3 GCN layers and our supervised VGAE architecture (`--use-vgae`), with an intermediate dimension in the graph layers of 256, and with batch normalisation between the graph layers. Furthermore, the  training command specifies a maximum atomic number of 35 and the maximum number of atoms in a molecule of 48 for this dataset, as well as the fact that this is a regression task and that this model is trained on a CPU (`--no-use-cuda`). Furthermore, the command specifies certain attributes specific to the dataset, such as the target label (column in the `.csv` file) to predict, in this case 'DR', the name of the SMILES column, in this case `neut-smiles`, and the name of the column containing each molecule's/compound's ID, in this case `CID`. This last option can be set to any column containing unique, identifying information (for example, SMILES for a curated dataset).

The model above corresponds to a 'base' (non-augmented) model that does not use low-fidelity information in any way. To augment the model with such data, the following arguments must be added:

```
--lbl-or-emb lbl --auxiliary-dim 1 --train-auxiliary-data-column-name <TRAIN-COL> --eval-auxiliary-data-column-name <EVAL-COL>
```

In this case, we want to augment using labels, so the auxiliary dimension is 1. We also provide the name of the columns containing the low-fidelity labels in the train/validation/test `.csv` files. The low-fidelity SD label is included by default in all DR files obtained from the MF-PCBA collection. Note that the data columns given by `--train-auxiliary-data-column-name` and `--eval-auxiliary-data-column-name` must be present for all molecules in all 3 files: train, validation, and test. The two options can be used to perform hybrid experiments where one type of augmentation is used during training, and another during evaluation. For normal, non-hybrid augmentation experiments, the two options should be set to the same value.


### 2. Augmenting with low-fidelity embeddings or predicted labels
Augmenting with low-fidelity embeddings or predicted labels requires training a separate supervised VGAE or vanilla GNN model exclusively on the entirety of the low-fidelity data, then extracting the corresponding embeddings/predictions.

#### a. Training a low-fidelity model
This task is handled with the Python script `train_low_fidelity.py`, which is very similar in usage to `train_high_fidelity.py`. An example:

```
python -m multifidelity_gnn.train.train_low_fidelity --data-path example_data/SD/SD.csv --low-fidelity-label SD --node-latent-dim 50
--graph-latent-dim 256 --out-dir out_SD --smiles-column neut-smiles --max-atomic-number 53 --max-num-atoms-in-mol 124
--readout set_transformer --id-column CID --monitor-loss-name train_total_loss --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --num-epochs 1 --gnn-intermediate-dim 256 --use-cuda --logging-name AID1445_LF_ST --batch-size 512
--dataloader-num-workers 12 --set-transformer-hidden-dim 1024 --set-transformer-num-heads 16 --set-transformer-num-sabs 2
--set-transformer-dropout 0.0
```

By default, CUDA is used due to the large number of low-fidelity training points. The example here uses the Set Transformer readout (`set_transformer`) as it is the only one that is effective in this learning task. Alternatively, the readout can be set to `global_add_pool`, and all the `--set-transformer-` options should be dropped.


#### b. Extracting the molecular/graph embeddings and the predicted low-fidelity values
Once the model has finished training, it can be loaded up from the latest checkpoint to produce graph embeddings and predicted labels for all the low-fidelity molecules in the dataset.

This task is covered by the same `train_low_fidelity.py` script, but a checkpoint path (`--ckpt-path`) must be specified, and the boolean option `--load-ckpt-and-generate-embs` must be enabled:

```
python -m multifidelity_gnn.train.train_low_fidelity --data-path example_data/SD/SD.csv --low-fidelity-label SD --node-latent-dim 50
--graph-latent-dim 256 --out-dir out_SD --smiles-column neut-smiles --max-atomic-number 53 --max-num-atoms-in-mol 124
--readout set_transformer --id-column CID --monitor-loss-name train_total_loss --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --num-epochs 1 --gnn-intermediate-dim 256 --use-cuda --logging-name AID1445_LF_ST --batch-size 512
--dataloader-num-workers 12 --set-transformer-hidden-dim 1024 --set-transformer-num-heads 16 --set-transformer-num-sabs 2
--set-transformer-dropout 0.0 --ckpt-path <CKPT-PATH> --load-ckpt-and-generate-embs
```

Note that the same settings that were used in `train_sd.py` must be used here as well, with only the last two options being new. The above script saves three files in the `out-dir`:

- `low_fidelity_graph_embeddings.npy` -- the graph embeddings for each low-fidelity molecule in a NumPy array
- `low_fidelity_predictions.npy` -- the predicted SD activity values for each low-fidelity molecule in a NumPy array
- `low_fidelity_true.npy` -- the actual/ground truth low-fidelity activity values (same as in the original low-fidelity `.csv` file) in a NumPy array

#### c. Joining the embeddings/predicted low-fidelity values to the `.csv` files
Assuming that the corresponding data frame (`df`) and the embeddings NumPy array (`arr`) have been loaded, the format expected by the training code can be performed by the simple command:
```
df['Embeddings'] = arr.tolist()
```
and for the predicted labels simply:
```
df['PredLF'] = arr
```

Generally, we compute the molecular embeddings and the predicted labels for the entire low-fidelity dataset and then add them to the corresponding low-fidelity data frame. Alternatively, any data set (.csv) file can be provided to `train_low_fidelity.py` with a corresponding checkpoint file.

#### d. Transition to high-fidelity data and split into train/validation/test sets
Assuming that at the previous step the low-fidelity embeddings and predictions were appended to the low-fidelity data frame, a further step might be required in order to make this information available in the high-fidelity data frame. If a high-fidelity data frame (`df_hq`) without any low-fidelity information and a low-fidelity data frame (`df_lq`) from the previous step are available, then one can simply perform a merge operation:

```
df_hq_with_lq = df_hq.merge(df_lq, on='CID')
```
Here, `CID` is a unique identifier present for every molecule.

If the low-fidelity embeddings and predicted labels were generated directly for the high-fidelity subset of molecules, then this merge step is not necessary, as the arrays can be simply appended to the high-fidelity data frame as in the previous step.

Finally, a high-fidelity data frame with all the necessary information can be split into train, validation, and test sets using the code available in the MF-PCBA repository for the drug discovery data, or by using the information provided in this repository for QMugs.

#### e. Training a high-fidelity model augmented with low-fidelity information
With the appropriate data (`.csv` files containing the embeddings and/or predicted labels), the training command for `train_high_fidelity.py` requires minimal changes:

```
--lbl-or-emb emb  --auxiliary-data-column-name Embeddings  --auxiliary-dim 256
```

Since the graph embedding dimension was set to 256 in the low-fidelity modelling step with Set Transformer readouts (`--graph-latent-dim 256`).


### 3. Shallow (RF/SVM) models
The training scripts for RF and SVM perform hyperparameter optimisation by default and will use all the available CPU cores for the algorithms that allow this option. The scripts also assume that the provided train/validation/test `.csv` files are formatted according to the examples discussed above. So far, we have used the shallow models only for the drug discovery data. As a result, the naming scheme for the arguments reflects this (SD/DR instead of low-fidelity/high-fidelity).

The training script will save the trained models, predictions and metrics in a provided save directory. RF and SVM are provided in the same script (`train_shallow_high_fidelity.py`).

An RF example:

```
python -m multifidelity_gnn.train.train_shallow_high_fidelity --data-path example_data/DR --smiles-column neut-smiles
--DR-label DR --fp-or-pc fp --rf-or-svm rf --task-type regression --save-path out_shallow
```

The selection of RF or SVM models is made using the `--rf-or-svm` option (possible values: `rf`, `svm`). In addition, the used input type can be set to fingerprints (`--fp-or-pc fp`) or RDKit Physical-Chemical descriptors (`--fp-or-pc pc`).

As for the deep learning models, the shallow models can be augmented with low-fidelity labels:

```
python -m multifidelity_gnn.train.train_shallow_high_fidelity --data-path example_data/DR --smiles-column neut-smiles
--DR-label DR --fp-or-pc fp --rf-or-svm rf --task-type regression --save-path out_shallow
--lbl-or-emb lbl --train-SD-label SD --eval-SD-label SD
```

or with low-fidelity embeddings:

```
python -m multifidelity_gnn.train.train_shallow_high_fidelity --data-path example_data/DR --smiles-column neut-smiles
--DR-label DR --fp-or-pc fp --rf-or-svm rf --task-type regression --save-path out_shallow
--lbl-or-emb emb --SD-EMBS-label STEmbeddings
```

## Pre-train on low-fidelity and fine-tune on high-fidelity
The first step for the supervised pre-training/fine-tuning workflow is to train a low-fidelity model as usual, for example using the same command as above:

```
python -m multifidelity_gnn.train.train_low_fidelity --data-path example_data/SD/SD.csv --low-fidelity-label SD --node-latent-dim 50
--graph-latent-dim 256 --out-dir out_SD --smiles-column neut-smiles --max-atomic-number 53 --max-num-atoms-in-mol 124
--readout set_transformer --id-column CID --monitor-loss-name train_total_loss --use-vgae --num-layers 3 --conv GCN
--use-batch-norm --num-epochs 1 --gnn-intermediate-dim 256 --use-cuda --logging-name AID1445_LF_ST --batch-size 512
--dataloader-num-workers 12 --set-transformer-hidden-dim 1024 --set-transformer-num-heads 16 --set-transformer-num-sabs 2
--set-transformer-dropout 0.0
```

This saves checkpoints (`.ckpt`) for every epoch in the out directory (`CKPT_PATH`). We will use this model as a base for fine-tuning:

```
python -m multifidelity_gnn.train.train_fine_tune_high_fidelity --high-fidelity-data-path example_data/DR --high-fidelity-label DR
--node-latent-dim 50 --graph-latent-dim 256 --out-dir <OUT_DIR> --smiles-column neut-smiles --max-atomic-number 53
--max-num-atoms-in-mol 124 --readout set_transformer --id-column CID --use-vgae --num-layers 3 --conv GCN --use-batch-norm --gnn-intermediate-dim 256 --use-cuda --logging-name <NAME> --batch-size 512 --dataloader-num-workers 12
--set-transformer-hidden-dim 1024 --set-transformer-num-heads 16 --set-transformer-num-sabs 2 --set-transformer-dropout 0.0
--low-fidelity-ckpt-path <CKPT_PATH> --freeze-vgae
```

Note that when using the Set Transformer readout the recommended fine-tuning strategy is to freeze the VGAE layers (`--freeze-vgae`). For low-fidelity models using the sum readout, the `--no-freeze-vgae` option should be used instead since the readout function is not learnable.


## Multi-fidelity state embedding (MFSE)
To run the MFSE algorithm with only high-fidelity data:

```
python -m mfse.train  --high-fi-train-data-path example_data/DR/train.csv
--high-fi-val-data-path example_data/DR/validate.csv  --high-fi-test-data-path example_data/DR/test.csv
--high-fi-batch-size 32 --high-fi-target-label DR --just-high-fi --max-atomic-number 35 --edge-dim 13
--node-latent-dim 50 --edge-latent-dim 50 --fidelity-embedding-dim 16 --use-set2set --smiles-column neut-smiles
--id-column CID --out-dir <OUT_PATH> --logging-name testing-high-fi --task-type regression
--dataloader-num-workers 2 --use-cuda --lr 0.0001 --num-layers 3 --no-use-batch-norm
```

To jointly use the low-fidelity data:
```
python -m mfse.train  --high-fi-train-data-path example_data/DR/train.csv
--high-fi-val-data-path example_data/DR/validate.csv  --high-fi-test-data-path example_data/DR/test.csv
--high-fi-batch-size 32 --high-fi-target-label DR --low-fi-data-path example_data/SD/SD.csv
--low-fi-target-label SD --low-fi-batch-size 256 --no-just-high-fi --max-atomic-number 35 --edge-dim 13
--node-latent-dim 50 --edge-latent-dim 50 --fidelity-embedding-dim 16 --use-set2set --smiles-column neut-smiles
--id-column CID --out-dir <OUT_PATH> --logging-name testing-low+high-fi --task-type regression
--dataloader-num-workers 8 --use-cuda --lr 0.0001 --num-layers 3 --no-use-batch-norm
```


## SchNet with more than 2 fidelities
We have adapted the source code from the paper **Modelling local and general quantum mechanical properties with attention-based pooling** (hosted at [https://github.com/davidbuterez/attention-based-pooling-for-quantum-properties](https://github.com/davidbuterez/attention-based-pooling-for-quantum-properties)) to support multi-fidelity learning with more than 2 fidelities.

To train a baseline model without any augmentations:

```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH>
```

To train a model augmented with labels (for example ZINDO):

```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH> --lbl-or-emb lbl --include zindo --aux-dim 1
```

With Set Transformer (ST) embeddings from one of the lower fidelities:
```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH> --lbl-or-emb emb --include pbe0 --emb-type st --aux-dim 256
```


Or sum embeddings from one of the lower fidelities:
```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH> --lbl-or-emb emb --include pbe0 --emb-type sum --aux-dim 256
```

Labels from both of the lower fidelities:
```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH> --lbl-or-emb lbl --include both --aux-dim 1
```

Or embeddings from both of the lower fidelities:
```
python -m schnet_multiple_fidelities.train --dataset QM7 --readout sum --lr 0.0001 --batch-size 128 --use-cuda
--target-id 4 --schnet-hidden-channels 256 --schnet-num-filters 256 --schnet-num-interactions 8 --random-seed 23887
--out-dir <OUT_PATH> --lbl-or-emb emb --include both --emb-type st --aux-dim 256
```


## Requirements/installation
The main dependencies are PyTorch, PyTorch Geometric, PyTorch Lightning, and RDKit. Certain steps also require `pandas`, `numpy`, `scipy`, `sklearn`, and `tqdm`. The install time is determined in large part by the quality of the internet connection, but should take less than 30 minutes on a normal computer.

The latest releases of the above work with our code (tested up to PyTorch 2.1 and PyTorch Geometric 2.3). For example:

1. Install a CUDA-enabled version of PyTorch
 ```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
2. Install PyTorch Geometric
```
conda install pyg -c pyg
```
3. Install PyTorch Lightning
```
pip install "pytorch-lightning==1.9.5"
```
4. Install RDKit
```
conda install rdkit -c conda-forge
```

Note that different versions of RDKit might produce slightly different results when filtering the datasets or when computing Physical-Chemical descriptors.

An example conda environment file is provided in this repository (`env.yaml`).

### Tested versions
The code was primarily developed and tested on a computer running Ubuntu 21.10, PyTorch 1.10.1 (with CUDA 11.3), PyTorch Geometric 2.0.3, PyTorch Lightning 1.5.7, and RDKit 2021.09.3.

The code was also tested on a different Linux platform with PyTorch 1.11.0 (with CUDA 11.3), PyTorch Geometric 2.1.0, PyTorch Lightning 1.6.0, and RDKit 2021.09.4.

We have also successfully run the code on PyTorch 1.13.1, a nightly version of PyTorch (1.14.0.dev20221026), PyTorch Geometric 2.1.0 (**installed from source, not from pip or conda**, as well as from conda), PyTorch Lightning 1.7.7 and 1.9.4, and RDKit 2022.09.1. We have also tested the code on macOS Ventura (13.0.1) using only the CPU.

### Tested hardware
The code was tested on an NVIDIA GeForce RTX 3090 24GB GPU (running under Ubuntu 21.10, with driver version 510.73.05), NVIDIA Tesla V100 16GB and 32GB GPUs, and an Apple M1 Max chip (CPU only).
