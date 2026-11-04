# ECE1512 Project A: ABMIL and Confidence Calibration

This project uses Attention-Based Multiple Instance Learning (ABMIL) to classify whole-slide images (WSIs). The model uses the patch features supplied with the course dataset. The experiments cover CAMELYON16 (C16), CAMELYON17 (C17), and BRACS.

The notebook includes the completed model, training code, test evaluation, and attention plots. Step 9 presents two research proposals. Proposal 1 examines how important high-attention patches are to a prediction. Proposal 2 uses temperature scaling to adjust prediction confidence. This project implements and evaluates Proposal 2. The report contains the literature review and the full discussion of the results.

## Files and Saved Results

| File or folder | Contents |
|---|---|
| `ECE1512_ProjectA_Student.ipynb` | Completed model, training, evaluation, attention plots, two proposals, and temperature scaling |
| `dataloader.py` | Supplied code for reading HDF5 features, splitting datasets, and forming batches |
| `utils.py` | Supplied code for configurations, random seeds, feature dimensions, and classification metrics |
| `wsi_visualization.py` | Supplied attention-plotting code with an added option to save figures |
| `config/` | Dataset settings supplied with the starter package |
| `dataset_csv/` | C17 and BRACS metadata supplied with the starter package |
| `requirements.txt` | Python packages listed in the starter package |
| `environment_projectA.txt` | Package versions recorded from the Windows experiment environment |
| `results/ABMIL/` | Baseline models, settings, data splits, training records, test results, and figures |
| `results/TemperatureScaling/` | Fitted temperatures, saved model scores, predictions, metrics, and figures |

The `dataset_csv/` folder needs both supplied CSV files. The data-loading code uses them to build the C17 and BRACS splits.

Each baseline has its own folder: `C16_seed42`, `C17_seed42`, or `BRACS_seed42`. Each folder contains the following files.

| File | Contents |
|---|---|
| `best_model.pt` | Model weights from the selected epoch |
| `config.json` | Settings used for that run |
| `splits.json` | Exact slide lists for training, validation, and testing |
| `history.csv` | Loss and metrics recorded after each epoch |
| `training_summary.json` | Training settings, selected epoch, and software information |
| `test_metrics.json` | Final baseline test results |
| `loss_curves.png` | Training and validation loss curves |

The C16 folder also contains `test_067_attention.png` and `test_068_attention.png`.

Each temperature-scaling folder contains `experiment.json`, `logits.npz`, `predictions.csv`, `reliability_bins.csv`, and `reliability_diagram.png`. The file `results/TemperatureScaling/summary.csv` combines the results from all three datasets. Each `logits.npz` stores validation and test scores, labels, and slide IDs. These saved scores support later metric checks without another model run.

## Environment and Local Launch

The experiments used the following environment.

| Item | Recorded value |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |

A Python 3.12 environment can run the project. The following commands install the listed packages and start JupyterLab. The terminal should use the project root as its working directory.

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m pip check
python -m jupyterlab
```

The notebook kernel should use the same Python environment. The notebook installation cell can stay unrun when the packages are already installed.

GPU execution needs a PyTorch build and NVIDIA driver that support the GPU. The notebook prints the PyTorch version, CUDA availability, and GPU name. A standard package installation may select a different PyTorch build. The file `environment_projectA.txt` records the original package versions. It includes Windows-specific packages, so another operating system may need a different environment setup.

The existing local Windows setup uses these commands.

```powershell
Set-Location "D:\ChromeDownload\1512\ProjectA\Project_A_Code"
& "D:\ChromeDownload\1512\.venv\Scripts\python.exe" -m jupyterlab
```

These Windows paths belong to the original local setup. Another computer can use its own project folder and Python environment. JupyterLab should start in the project root because the configuration files use relative paths.

## Data Preparation

The pre-extracted features are available from the [course Zenodo record](https://zenodo.org/records/23130543). The [course Hugging Face dataset](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature) provides another download source and the slides used for visualization.

| Dataset | Feature path | Classes | Train / validation / test |
|---|---|---:|---:|
| C16 | `feature/C16/patch_feats_pretrain_medical_ssl.h5` | 2 | 242 / 27 / 129 |
| C17 | `feature/C17/patch_feats_pretrain_medical_ssl.h5` | 4 | 269 / 30 / 200 |
| BRACS | `feature/BRACS/patch_feats_pretrain_medical_ssl.h5` | 7 | 389 / 62 / 86 |

The three HDF5 files have the same filename. Their separate dataset folders keep the files apart. A different storage location requires a matching `data_dir` in the dataset configuration.

The original downloaded archives matched these MD5 values.

| Archive | MD5 |
|---|---|
| `C16.zip` | `f2a105a5ab17186fcf59e976870d5f1e` |
| `C17.zip` | `f98b87a8527bb1f54969c6a2d939e49c` |
| `BRACS.zip` | `c521971286cafe005e081d213fd17ed8` |

Step 8 also needs `test_067.tif`, `test_068.tif`, and `test_068.xml` in the `slide/` folder. Training and temperature scaling use the HDF5 features and do not need these original slide images. The source datasets provide the license and attribution terms for the downloaded data.

## View or Check the Saved Results

The saved notebook and result files show the recorded experiments. Readers can inspect the metric JSON files, training CSV files, figures, and calibration summary directly.

The training and calibration cells stop when their output folders already exist. This check protects the saved experiments. A reader who only wants to inspect the results does not need to select **Run All**.

### Recalculate Calibration Metrics

The saved logits and fitted temperatures allow the metrics to be checked without the HDF5 features or original slide images. The required Python packages still need to be installed.

1. You can run the Step 0 imports and project-directory setup cells.
2. Step 9.1 defines the calibration functions.
3. Step 9.4 reads the saved logits and temperatures. It recalculates the metrics and updates the experiment JSON files and summary CSV. It does not train ABMIL or fit new temperatures.
4. Step 9.5 redraws the reliability diagrams. It also needs the saved `reliability_bins.csv` files.

The `before` and `after` fields in `experiment.json` contain the reported metrics. The summary CSV contains the same comparison. The `auc_numerical_audit` field keeps earlier AUC values for reference when that field is present. These values are separate from the reported `before` and `after` results.

### Regenerate C16 Attention Figures

The attention figures need the C16 HDF5 features and the three visualization files in `slide/`.

1. You can run the Step 0 imports, project-directory setup, and configuration-loading cells.
2. Step 2 defines the model classes.
3. Step 3 defines the device and the initial model objects.
4. The two Step 8 code cells load the saved C16 model and generate the figures.

Step 8 reads the saved C16 settings and test split. The earlier `DATASET` selection does not change which model Step 8 uses. This route replaces the two saved attention PNG files.

### Saved Notebook Outputs

The notebook keeps the original baseline training and test outputs. The latest partial run reloaded the model definitions and reran Steps 9.1-9.5 with the saved baseline checkpoints. That run did not retrain the baseline models. Step 9.3 now includes the three dataset results and the six-row comparison table. Step 9.4 checks the final metrics again.

## Repeat the Full Experiments

A new complete run needs empty output locations. The existing `results/ABMIL/` and `results/TemperatureScaling/` folders should move to a backup outside the repository before that run. The backup keeps the original results available for comparison.

1. You can set `PROJECT_DIR = "./"` and `DATASET = "C16"` in Step 0.
2. The setup cells and Steps 1-7 train and evaluate the C16 baseline. The reported experiment uses all 30 epochs. Step 6 saves the selected model, and Step 7 saves its test results.
3. The same process trains C17 and BRACS after the dataset selection changes. Each change also requires the configuration, Dataset, and DataLoader cells to run again. A kernel restart between datasets can keep the runs separate.
4. Step 8 generates the C16 attention figures from its saved model.
5. Steps 9.1 and 9.2 define and check the calibration functions. Step 9.3 then fits and evaluates one temperature for each dataset. This step needs all three completed baselines.
6. Steps 9.4 and 9.5 check the final metrics and save the reliability figures. The notebook should be saved with these outputs.

Step 6 resets the seed and creates a fresh model for each training run. Step 9.3 uses validation data to fit the temperature and test data to evaluate it.

A `FileExistsError` in Step 6 or Step 9.3 means that the output folder already exists. A stopped run may leave a partial folder. That folder needs a backup and a new location before the same experiment starts again. Different hardware or software versions may produce small numerical differences.

## Baseline Settings and Results

The model maps each 384-dimensional patch feature to 256 dimensions. The gated-attention module uses an attention dimension of 128. The model combines the patch features and sends the result to a linear classifier.

The training settings are listed below.

| Setting | Value |
|---|---|
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Epochs | 30 |
| Batch size | 1 |
| Dropout | 0 |
| Random seed | 42 |

The selected epoch has the highest validation macro-AUROC + macro-F1. A later epoch replaces it only when the score is higher. A tied score keeps the earlier selected epoch. The test metrics do not select the model.

The model does not use the padding mask returned by the data loader. Batch size 1 avoids padded patches when slides contain different numbers of patches.

| Dataset | Best epoch | Test accuracy (%) | Macro-AUROC | Macro-F1 | Test loss |
|---|---:|---:|---:|---:|---:|
| C16 | 3 | 88.3721 | 0.9224 | 0.8714 | 0.2700 |
| C17 | 22 | 72.5000 | 0.8090 | 0.4815 | 0.9473 |
| BRACS | 3 | 46.5116 | 0.7858 | 0.3716 | 1.4110 |

## Temperature Scaling and Results

Temperature scaling keeps the trained ABMIL weights fixed. The code fits one positive temperature, T, for each dataset. LBFGS adjusts T to lower validation NLL. The allowed temperature range is 0.01 to 100.

The adjusted probabilities are `softmax(logits / T)`. A positive T keeps the predicted class unchanged. Accuracy and macro-F1 therefore stay the same. Multiclass one-vs-rest AUROC can still change because the probability ranking for an individual class can change across slides.

The binary AUROC calculation uses the difference between the two class scores. The multiclass calculation uses one-vs-rest log-odds scores for each class. These scores avoid extra ranking ties caused by rounded probabilities. Positive temperature scaling keeps binary AUROC unchanged.

| Dataset | Temperature | Test NLL before -> after | Brier before -> after | ECE (%) before -> after |
|---|---:|---:|---:|---:|
| C16 | 0.093815 | 0.2700 -> 1.7704 | 0.1564 -> 0.2086 | 3.27 -> 10.93 |
| C17 | 1.395575 | 0.9473 -> 0.8048 | 0.4259 -> 0.4099 | 11.21 -> 6.45 |
| BRACS | 1.179234 | 1.4110 -> 1.4073 | 0.6486 -> 0.6474 | 3.89 -> 6.30 |

NLL is the mean cross-entropy across slides. The Brier score adds the squared probability errors across classes and then averages the result across slides. This definition does not divide by the number of classes.

ECE compares confidence with accuracy in five equal-width bins. Each slide enters a bin through its highest class probability. The first bin includes both endpoints. Each later bin excludes its lower endpoint and includes its upper endpoint. The saved ECE values use a scale from 0 to 1. The figures and table above show ECE as a percentage.

Lower NLL, Brier score, and ECE indicate better results for those metrics. C17 improves on all three metrics. C16 becomes worse on all three. BRACS has slightly lower NLL and Brier score, but its ECE increases.

## Limitations and Code Sources

- The experiments use one random seed. The project uses supplied patch features and does not train the feature extractor.
- The validation sets are small. The same validation data support both checkpoint selection and temperature fitting. The selected C16 baseline correctly classifies all 27 validation slides. Temperature fitting then produces sharper probabilities, while its test calibration becomes worse.
- The supplied C17 code splits training and validation data by slide. The saved training and validation sets share 27 patients. The test patients do not overlap either set. The test slides come from centers 3 and 4, while the training and validation slides come from centers 0, 1, and 2. The comparison keeps this supplied split. This C17 validation split is not patient-independent. Its test set is not the official challenge test partition.
- The attention colors show percentiles within each slide. They do not show patch tumor probabilities. An attention map alone does not prove which patches cause the prediction. The visualization treats patch coordinates as level-0 top-left positions. It assumes a patch width and height of 512 pixels because the HDF5 files do not provide patch-size metadata. The normal-slide example has no supplied XML annotation. These plots provide visual examples rather than a measured localization score.
- The instructor supplied the notebook, data utilities, configurations, metadata, and original visualization code. This project completes the TODOs and adds experiment saving, calibration, and figure saving. Some unchanged configuration comments say "Project B". The experiments here belong to Project A.

## Repository Contents

The repository should include the notebook, Python files, configurations, dataset metadata CSV files, dependency information, this README, and the saved experiment results. The baseline and calibration results have separate folders.

The `.gitignore` excludes Python environments, caches, downloaded archives, HDF5 features, and raw TIF/XML slide data. The data links and paths above explain how to obtain those inputs. Local backups and review ZIP files should stay outside the repository. The Git staging list shows which files will be uploaded. Ignore rules do not remove files that Git already tracks.

The report should include the final repository link. The report and code also need their separate reproducibility checklists.

## References

1. M. Ilse, J. M. Tomczak, and M. Welling. Attention-based Deep Multiple Instance Learning. ICML, 2018. [Paper](https://proceedings.mlr.press/v80/ilse18a.html).
2. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML, 2017. [Paper](https://proceedings.mlr.press/v70/guo17a.html).
3. L. Ye. WSI Feature. Zenodo, 2026. [Dataset record](https://zenodo.org/records/23130543).
