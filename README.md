# MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction

Official pytorch code for Mantra: Memory augmented networks for multiple trajectory prediction - CVPR2020

[MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction" by Francesco Marchetti, Federico Becattini, Lorenzo Seidenari, and Alberto Del Bimbo.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.pdf)

![Multiple trajectory prediction. Blue: past, red: futures.][gif]

[gif]: https://github.com/Marchetz/MANTRA-CVPR20/blob/master/mantra.gif "MANTRA"

## Installation
To install the required packages, in a Python 3.6 environment just execute the following: 
```bash
pip install -r requirements.txt
```

## Dataset
We provide a dataloader for the KITTI dataset in *dataset_invariance.py*. The dataloader yields samples of (past, future) trajectories paired with a semantic map of the surrounding scene.

## Training
To train MANTRA, first it is necessary to train the autoencoder, then to train the writing controller and finally to train the Iterative
Refinment Module (IRM).
Trainings can be monitored using tensorboard, logs are stored in the folder *runs/(runs-pretrain/runs-createMem/runs-IRM)*.
In the *pretrained_model* folder there are pretrained models of the different components (autoencoder, writing controller, MANTRA).

**Note:** Two versions of training scripts are available:
- Original scripts (`train_ae.py`, `train_controllerMem.py`, `train_IRM.py`) for KITTI dataset
- V2 scripts (`train_ae_v2.py`, `train_controllerMem_v2.py`, `train_IRM_v2.py`) for custom 3D trajectory data

### Training encoder-decoder model (autoencoder)

#### Original version (KITTI):
```bash
python train_ae.py
```
The autoencoder can be trained with the **train_ae.py** script. train_ae.py calls trainer_ae.py
The model will be saved into the folder *test/[current_date]*.
A pretrained model can be found in *pretrained_models/model_AE/*

#### V2 version (Custom 3D data):
```bash
python train_ae_v2.py --batch_size 32 --learning_rate 0.0001 --max_epochs 600 --past_len 21 --future_len 20
```
The **train_ae_v2.py** script calls trainer_ae_v2.py and uses PyTorch Lightning for training.
Checkpoints are saved in *training/training_ae/* with timestamped folders.

**Optional arguments:**
- `--batch_size`: Number of samples per batch (default=32)
- `--learning_rate`: Learning rate for optimizer (default=0.0001)
- `--max_epochs`: Maximum number of training epochs (default=600)
- `--past_len`: Length of past trajectory in timesteps (default=21)
- `--future_len`: Length of future trajectory in timesteps (default=20)
- `--dim_embedding_key`: Dimension of embedding key (default=48)
- `--info`: Additional info for naming the training run

### Training writing controller

#### Original version (KITTI):
```bash
python train_controllerMem.py --model pretrained_autoencoder_model_path
```
The writing controller for the memory with autoencoder can be trained with **train_controllerMem.py**.
train_controllerMem.py calls trainer_controllerMem.py.
The path of a pretrained autoencoder model has to be passed to the script (it defaults to the pretrained model we provided).
A pretrained model (autoencoder + writing controller) can be found in *pretrained_models/model_controller/*

#### V2 version (Custom 3D data):
```bash
python train_controllerMem_v2.py --model_ae autoencoder_20steps_checkpoint/best-ae-epoch=XX-val_eucl_mean=X.XXXX.ckpt
```
The **train_controllerMem_v2.py** script calls trainer_controllerMem_v2.py and trains the memory writing controller.
You must provide the path to a trained autoencoder checkpoint.
Checkpoints are saved in *training/training_controller/* with timestamped folders.

**Optional arguments:**
- `--batch_size`: Number of samples per batch (default=32)
- `--learning_rate`: Learning rate for optimizer (default=0.0001)
- `--max_epochs`: Maximum number of training epochs (default=600)
- `--past_len`: Length of past trajectory in timesteps (default=21)
- `--future_len`: Length of future trajectory in timesteps (default=20)
- `--preds`: Number of predictions to generate (default=5)
- `--dim_embedding_key`: Dimension of embedding key (default=48)
- `--model_ae`: Path to pretrained autoencoder checkpoint (required)
- `--info`: Additional info for naming the training run

### Training Iterative Refinement Module (IRM)

#### Original version (KITTI):
```bash
python train_IRM.py --model pretrained_autoencoder+controller_model_path
```
train_IRM.py calls trainer_IRM.py
The script trains the IRM module that generates the final prediction based on the decoded trajectory and the context map.
The paths of a pretrained autoencoder with writing controller model and populated memories have to be passed to the script (it defaults to the
pretrained models we provided).
A pretrained MANTRA model can be found in *pretrained_models/model_complete/*

#### V2 version (Custom 3D data):
```bash
python train_IRM_v2.py --model training/training_controller/YYYY-MM-DD_HH-MM-SS/checkpoints/model_controller-epoch=XX-val_loss=XX.XXXX.ckpt
```
The **train_IRM_v2.py** script calls trainer_IRM_v2.py and trains the complete MANTRA model with IRM.
You must provide the path to a trained controller+autoencoder checkpoint.
Checkpoints are saved in *training/training_IRM/* with timestamped folders.

**Optional arguments:**
- `--batch_size`: Number of samples per batch (default=32)
- `--learning_rate`: Learning rate for optimizer (default=0.0001)
- `--max_epochs`: Maximum number of training epochs (default=600)
- `--past_len`: Length of past trajectory in timesteps (default=21)
- `--future_len`: Length of future trajectory in timesteps (default=20)
- `--preds`: Number of predictions to generate (default=5)
- `--dim_embedding_key`: Dimension of embedding key (default=48)
- `--model`: Path to pretrained controller checkpoint (required)
- `--saved_memory`: Use saved memory bank (default=True)
- `--saveImages`: Save qualitative examples in tensorboard (default=True)
- `--info`: Additional info for naming the training run

### Training Pipeline Example (V2):
```bash
# Step 1: Train autoencoder
python train_ae_v2.py --max_epochs 100

# Step 2: Train writing controller (use best autoencoder checkpoint)
python train_controllerMem_v2.py --model_ae autoencoder_20steps_checkpoint/best-ae-epoch=XX-val_eucl_mean=X.XXXX.ckpt --max_epochs 100

# Step 3: Train IRM (use best controller checkpoint)
python train_IRM_v2.py --model training/training_controller/YYYY-MM-DD_HH-MM-SS/checkpoints/model_controller-epoch=XX-val_loss=XX.XXXX.ckpt --max_epochs 100
```


## Test
```bash
python test.py --model pretrained_complete_model_path --withIRM True/False --saved_memory True/False
```
test.py calls evaluate_MemNet.py
This script generates metrics on the KITTI dataset using a trained models. We compute Average Displacement Error (ADE) and Final Displacement Error (FDE, also referred to as Error@K or Horizon Error).

### Command line arguments
```
    --cuda                         Enable/Disable GPU device (default=True).
    --batch_size                   Number of samples that will be fed to MANTRA in one iteration (default=32).
    --past_len                     Past length (default=20).
    --future_len                   Future length (default=40).
    --preds                        Number of predictions generated by MANTRA model (default=5)
    --model                        Path of pretrained model for the evaluation (default='pretrained_models/MANTRA/model_MANTRA')
    --visualize_dataset            The system saves (in *folder_test/dataset_train* and *folder_test/dataset_test*) all examples
                                   of dataset.
    --saved_memory                 The system chooses which memories will be used in evaluation.
                                   If True, it will be loaded memories from 'memories_path' folder.
                                   If False, new memories will be generated. pairs of past-future will be decided by writing controller of model.
    --memories_path                This path will be used only if saved_memory flag is True.
    --withIRM                      The model generates predictions with/without Iterative Refinement Module.
    --saveImages                   The system saves in test folder examples of dataset with prediction generated by MANTRA.
                                   If None, it doesn't save any qualitative examples but only quantitative results.
                                   If 'All', it saves all examples.
                                   If 'Subset', it saves examples defined in index_qualitative.py (hand picked most significant samples)
                                   (default=None)
    --dataset_file                 Name of json file cointaining the dataset (default='kitti_dataset.json')
    --info                         Name of evaluation. It will use for name of the test folder (default='')

```

## Evaluation & Inference (V2)

### Running Inference with Metrics
For custom 3D trajectory data, use **infer_w_metrics.py** to evaluate trained models and generate comprehensive metrics:

```bash
python infer_w_metrics.py --checkpoint training/training_IRM/YYYY-MM-DD_HH-MM-SS/checkpoints/model_IRM-epoch=XX-val_eucl_mean=X.XXXX.ckpt --test_data data/test_dataset --out_dir evaluation_results
```

This script will:
- Generate multi-modal trajectory predictions (3D)
- Compute Winner-Takes-All metrics (minADE, minFDE)
- Create 2D and 3D visualization plots
- Save error growth plots with uncertainty bands
- Generate ADE distribution histograms
- Export metrics to pickle files

**Command line arguments:**
```
    --checkpoint                   Path to trained IRM checkpoint (required)
    --test_data                    Directory containing test data in JSON format (required)
    --out_dir                      Output directory for plots and metrics (default='evaluation_results_full/')
    --batch_size                   Number of samples per batch (default=32)
    --num_plots                    Number of sample visualizations to save (default=100)
    --past_len                     Length of past trajectory in timesteps (default=21)
    --future_len                   Length of future trajectory in timesteps (default=20)
    --preds                        Number of trajectory predictions to generate (default=5)
    --dim_embedding_key            Dimension of embedding key (default=48)
```

### Aggregating Multiple Evaluation Results
To aggregate metrics across multiple test datasets:

```bash
python aggregate_metrics.py --results_dir evaluation_results_full --out_dir evaluation_results_aggregated
```

This script will:
- Crawl through all subdirectories to find `metrics.pkl` files
- Average metrics across all test sets
- Generate aggregated error growth plots with cross-dataset variation
- Create comparison plots showing per-dataset performance
- Export aggregated metrics to CSV files (timestep_metrics.csv, summary_metrics.csv, all_sample_ades.csv)
- Save aggregated metrics as pickle file

**Example workflow:**
```bash
# Evaluate on multiple datasets
python infer_w_metrics.py --checkpoint model.ckpt --test_data data/dataset1 --out_dir evaluation_results_full/dataset1
python infer_w_metrics.py --checkpoint model.ckpt --test_data data/dataset2 --out_dir evaluation_results_full/dataset2
python infer_w_metrics.py --checkpoint model.ckpt --test_data data/dataset3 --out_dir evaluation_results_full/dataset3

# Aggregate all results
python aggregate_metrics.py --results_dir evaluation_results_full --out_dir evaluation_results_aggregated
```


#### Citation

If you use our code or find it useful in your research, please cite the following paper:


<pre class='bibtex'>
@inproceedings{cvpr_2020,
 author = {Marchetti, Francesco and  Becattini, Federico and Seidenari, Lorenzo and Del Bimbo, Alberto},
 booktitle = {International Conference on Computer Vision and Pattern Recognition (CVPR)},
 publisher = {IEEE},
 title = {MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction},
 year = {2020}
}
</pre>

<pre class='bibtex'>
@ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
</pre>

#### License

![logo](logo-imra.png)

This source code is shared under the license CC-BY-NC-SA, please refer to the [LICENSE](LICENSE) file for more information.

This source code is only shared for R&D or evaluation of this model on user database.

Any commercial utilization is strictly forbidden.

For any utilization with a commercial goal, please contact [contact_cs](mailto:contact_cs@imra-europe.com) or [bendahan](mailto:bendahan@imra-europe.com)
