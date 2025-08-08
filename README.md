# TensorRF-Debloated 🎉

Welcome to **TensorRF-Debloated**, a lightweight, minimalistic implementation of TensoRF designed for ease of understanding, quick experimentation, and fast prototyping.

## 📦 Features

* **Debloated & Simple**: Stripped-down codebase focusing on core components.
* **Flexible Configs**: JSON-based experiments and training configurations.
* **Sparse Alpha Masking**: Efficient ray filtering via `<span>AlphaGridMask</span>`.
* **Two Shading Modes**: `<span>MLP_PE</span>` and `<span>MLP_Fea</span>` with positional & feature encoding.
* **Progressive Upsampling**: Seamless resolution scaling during training.
* **Easy Checkpointing**: Save and load full model + mask data with a single call.

## 🚀 Quickstart

### 1. Clone & Install

```
git clone https://github.com/ahsanjmirza/tensorrf-debloated.git
cd tensorrf-debloated
pip install -r requirements.txt
```

### 2. Prepare Dataset

Structure your dataset directory as follows:

```
experiments/<experiment_name>/
├── cam_images/      # input images
├── masks/           # optional foreground masks
├── train.json       # training metadata
├── val.json         # validation metadata
└── <experiment>.pth # checkpoints and outputs
```

Fill `<span>train.json</span>` and `<span>val.json</span>` similarly to the provided `<span>experiments/vishnu_nprog/</span>` folder.

### 3. Configure & Train

* Edit `<span>configs/vishnu_nprog.json</span>` (or copy to a new experiment).
* Run training:
  ```
  python trainer.py --config configs/vishnu_nprog.json
  ```

> By default, the non-progressive pipeline (`<span>nprog</span>`) is used. To enable progressive training, adjust `<span>progressive: true</span>` in your `<span>Training</span>` block.

### 4. Evaluate & Visualize

After training, rendered images are saved to:

```
experiments/<experiment_name>/eval/
```

Extract final models and snapshots:

```
ls experiments/<experiment_name>/eval/*.png
```

## 🗂️ File Structure

```
`tensorrf-debloated/`
├── **configs/**
│   └── `vishnu_nprog.json`    # Sample experiment config
├── **model/**
│   ├── **tensorial.py**       # Core TensoRF implementation
│   └── **utils/**
│       ├── `alpha_grid.py`    # Sparse alpha mask sampling
│       ├── `p_encode.py`      # Positional & feature encoding
│       ├── `raw2alpha.py`     # Converting raw sigma to alpha/weight
│       └── `render_module.py` # MLP-based shading modules
├── **experiments/**
│   └── `vishnu_nprog/`        # Provided example experiment
├── `trainer.py`               # Training & evaluation loop
├── `utils.py`                 # Helper functions (grid size calc)
└── `requirements.txt`         # Python dependencies
```

## ⚙️ Configuration Highlights

* **TensorBase**:
  * `<span>density_n_comp</span>`, `<span>appearance_n_comp</span>`, `<span>app_dim</span>` control SVD volume shape.
  * `<span>shadingMode</span>`: choose between `<span>MLP_PE</span>` (positional-encoded MLP) or `<span>MLP_Fea</span>` (feature-encoded MLP).
  * `<span>alphaMask_thres</span>`, `<span>rayMarch_weight_thres</span>`: thresholds for mask and sampling.
* **Training**:
  * `<span>n_iters</span>`, `<span>batch_size</span>`, `<span>upsamp_list</span>`: control iteration count, batch size, and upsampling epochs.
  * `<span>update_AlphaMask_list</span>`: epochs to refresh the alpha mask for pruning empty space.
  * `<span>lr_init</span>`, `<span>lr_basis</span>`, `<span>lr_decay_target_ratio</span>`: learning rates and decay schedule.
  * `<span>progressive</span>`: toggle frame-by-frame progressive batching.

## 🔧 Customization Tips

* **New Experiment**: Duplicate and modify a JSON config in `<span>configs/</span>`, update `<span>dataset_path</span>` and hyperparameters.
* **Fast Debugging**: Lower `<span>n_iters</span>` and `<span>batch_size</span>` for quick sanity checks.
* **Visualization**: Insert custom logging callbacks in `<span>Trainer.eval()</span>` to save depth maps or other metrics.

## 🎉 Acknowledgments

This implementation is inspired by the original [TensoRF](https://github.com/apchenstu/TensoRF) work. Hat tip to its authors for paving the way!
