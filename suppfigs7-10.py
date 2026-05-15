"""
generate_supplementary_figures.py
=================================

Produces every simulation-based supplementary figure required for the
npj Unconventional Computing revision:

    SF-IO       I-O firing curve         (Reviewer 1, Comment 1.1)
    SF-REPEAT   Repeatability (Jaccard)  (Reviewer 2, Comment 2.2)
    SF-NOISE    Input-noise robustness   (Reviewer 2, Comment 2.2)
    SF-DIST     Propagation-distance sweep (Reviewer 2, Comment 2.1)
    SF-CASC     Two-layer cascade demo   (Reviewer 2, Comment 2.6)

All physical conventions match `optical_net_train.py`:
    lambda = 635 nm,  8 um pitch,  480x480 grid,  z = 40 cm,
    8x8 superpixel,  top-1/3 I_sig,  I_RW = 2*I_sig,  8x8 average pool.

This script is SELF-CONTAINED: by default it trains a fresh single-layer
OpticalNet from scratch (same recipe as optical_net_train.py) and then
generates all five figures.  No prior checkpoint is required.  If you do
have a checkpoint, set TRAIN_FROM_SCRATCH = False below and point
CHECKPOINT_PATH at it -- this skips the ~30-60 minute training step.

The trained single-layer model is saved to './best_model_freshly_trained'
so subsequent re-runs can re-use it.

Each figure block is independent: comment out the ones you do not want
to re-run.  Outputs are saved into ./supplementary_figures/ as PNG + SVG
at 1200 dpi to match the existing rogue_wave_analysis.py conventions.

USER-EDITABLE THINGS THAT MUST BE CHECKED BEFORE RUNNING
--------------------------------------------------------
1. TRAIN_FROM_SCRATCH     True  -> ~30-60 min on a single GPU, but no
                                   checkpoint dependency
                          False -> load CHECKPOINT_PATH instead
2. BREAST_IMAGES_NPY      path to your cached 'breast_images.npy' file
                          (the same one rogue_wave_analysis.py uses).
                          If absent, the script falls back to medmnist.
                          For SF-NOISE, also need breast_labels.npy or
                          medmnist must be reachable.
3. NEURON_SITE_STRATEGY   how to pick the "representative neuron" for
                          the I-O curve (default: brightest pixel at
                          design amplitude).  Change if the brightest
                          pixel sits in a region you do not want to
                          highlight in the figure.
4. Number of test images for NOISE and DIST sweeps (TEST_SUBSET_SIZE).
   Set this to None to run on the full test set (slower).

Author: Bahadir Utku Kesgin (drafted with Claude)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils  # band-limited ASM — DO NOT MODIFY

try:
    import scienceplots  # noqa: F401  -- registers the styles
    plt.style.use(['science', 'nature'])
    HAS_SCIENCEPLOTS = True
except (ImportError, OSError):
    HAS_SCIENCEPLOTS = False
    print("[warn] scienceplots not available; falling back to default style.")


# ============================================================================
# 0.  GLOBAL CONFIGURATION
# ============================================================================
WAVELENGTH      = 635e-9
PIXEL_PITCH     = 8e-6
GRID            = 480
INPUT_SIZE      = 224
PROP_DISTANCE   = 40e-2
SUPERPIXEL      = 8
POOL            = 8

RESOLUTION      = [PIXEL_PITCH, PIXEL_PITCH]
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE           = torch.float32

CHECKPOINT_PATH    = 'best_model_freshly_trained'              # used only if TRAIN_FROM_SCRATCH = False
BREAST_IMAGES_NPY  = 'breast_images.npy'       # same file rogue_wave_analysis.py reads
USE_MEDMNIST_IF_NPY_MISSING = True             # set False to crash hard if .npy is absent

# -------- choose how to obtain the single-layer trained model ---------------
#   True  -> train a fresh single-layer OpticalNet from scratch in this script
#            (~30-60 min on a single GPU; same recipe as optical_net_train.py)
#   False -> load the checkpoint at CHECKPOINT_PATH
TRAIN_FROM_SCRATCH = False
SINGLE_LAYER_EPOCHS = 200    # bump to 400 to match optical_net_train.py exactly
SINGLE_LAYER_LR     = 1e-4
SINGLE_LAYER_BATCH  = 20
SINGLE_LAYER_PATIENCE = 50   # early-stopping patience in epochs

OUTPUT_DIR = Path('./supplementary_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot helpers
def cm_to_in(v): return v / 2.54

def add_label(ax, label, x=-0.05, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=7, fontweight='bold', va='top', ha='right')

def save_fig(fig, name):
    """Save both PNG (1200 dpi) and SVG (vector) versions."""
    p_png = OUTPUT_DIR / f"{name}.png"
    p_svg = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(p_png, dpi=1200, bbox_inches='tight')
    fig.savefig(p_svg, format='svg', bbox_inches='tight')
    print(f"  [saved] {p_png}")
    print(f"  [saved] {p_svg}")


# ============================================================================
# 1.  MODEL DEFINITIONS  (mirror of optical_net_train.py; kept self-contained
#                        so this script can be run independently)
# ============================================================================
def _build_source_field():
    """Gaussian source identical to A_field in optical_net_train.py."""
    spacewidth = GRID * PIXEL_PITCH
    x = torch.linspace(-spacewidth * 0.5, spacewidth * 0.5, GRID, dtype=DTYPE)
    y = torch.linspace(-spacewidth * 0.5, spacewidth * 0.5, GRID, dtype=DTYPE)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    beam_width = INPUT_SIZE * PIXEL_PITCH         # named "beam width" in training script
    x_fwhm = y_fwhm = beam_width / 2.355          # acts as 1/sqrt(norm), see training script
    A = torch.sqrt(
        (1.0 / (math.pi * x_fwhm * y_fwhm)) *
        torch.exp(
            -((X ** 2) / (2 * beam_width ** 2) +
              (Y ** 2) / (2 * beam_width ** 2))
        )
    )
    return A  # (GRID, GRID)


class PhasePattern(nn.Module):
    """Checkerboard superpixel phase mask (only the checkerboard branch used here)."""
    def __init__(self, height=INPUT_SIZE, width=INPUT_SIZE,
                 superpixel_size=SUPERPIXEL):
        super().__init__()
        self.height, self.width = height, width
        self.sp = superpixel_size
        grid_h = height // superpixel_size
        grid_w = width // superpixel_size
        self.phase_values = nn.Parameter(torch.rand(grid_h, grid_w) * 2 * math.pi)

    def forward(self):
        p = torch.repeat_interleave(self.phase_values, self.sp, dim=0)
        p = torch.repeat_interleave(p, self.sp, dim=1)
        return p[:self.height, :self.width]


class RogueWaveThreshold(nn.Module):
    """Soft top-1/3 rogue-wave threshold; matches optical_net_train.py."""
    def __init__(self, steepness=100.0):
        super().__init__()
        self.steepness = steepness

    def compute_threshold(self, intensity):
        B, H, W = intensity.shape
        flat = intensity.view(B, -1)
        k = max(1, flat.shape[1] // 3)
        top_vals, _ = torch.topk(flat, k, dim=1)
        I_sig = top_vals.mean(dim=1, keepdim=True)
        return (2.0 * I_sig).view(B, 1, 1)

    def forward(self, intensity, fixed_threshold=None):
        if fixed_threshold is None:
            threshold = self.compute_threshold(intensity)
        else:
            # broadcast-able to (B, 1, 1)
            threshold = fixed_threshold
        soft_mask = torch.sigmoid(self.steepness * (intensity - threshold))
        return soft_mask, threshold


class OpticalNet(nn.Module):
    """Single-layer model used in the manuscript."""
    def __init__(self, num_classes=2,
                 wavelength=WAVELENGTH,
                 input_size=(INPUT_SIZE, INPUT_SIZE),
                 target_size=(GRID, GRID),
                 superpixel_size=SUPERPIXEL,
                 pool_size=POOL,
                 steepness=100.0):
        super().__init__()
        self.wavelength = wavelength
        self.resolution = RESOLUTION
        self.target_size = target_size
        self.pad_h = (target_size[0] - input_size[0]) // 2
        self.pad_w = (target_size[1] - input_size[1]) // 2

        self.register_buffer('source_field', _build_source_field())
        self.phase_generator = PhasePattern(*input_size, superpixel_size)
        self.rogue_threshold = RogueWaveThreshold(steepness=steepness)
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

        feat = (target_size[0] // pool_size) * (target_size[1] // pool_size)
        self.classifier = nn.Linear(feat, num_classes)

    def pad_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
                     mode='constant', value=0)

    def propagate(self, input_amp, amplitude_scale=1.0, fixed_threshold=None,
                  return_intermediates=False):
        """
        Run the full forward pass with an optional amplitude scaling factor.
        Returns (logits, dict).  `amplitude_scale` multiplies the data
        amplitude before modulation, leaving the source Gaussian alone.
        """
        x = self.pad_input(input_amp)                              # (B,1,GRID,GRID)
        phi = self.pad_input(self.phase_generator())               # (1,1,GRID,GRID)
        modulated = self.source_field * (amplitude_scale * x) * torch.exp(1j * phi)
        U = utils.propagation_ASM(modulated, self.resolution,
                                  self.wavelength, PROP_DISTANCE)
        intensity = (torch.abs(U) ** 2).squeeze(1)                 # (B,GRID,GRID)
        soft_mask, threshold = self.rogue_threshold(intensity, fixed_threshold)
        pooled = self.pool(soft_mask.unsqueeze(1)).flatten(1)
        logits = self.classifier(pooled)
        if return_intermediates:
            return logits, {
                'intensity': intensity,
                'soft_mask': soft_mask,
                'threshold': threshold,
            }
        return logits

    # convenience
    def forward(self, input_amp):
        return self.propagate(input_amp)


def load_trained_model(path=CHECKPOINT_PATH, num_classes=2):
    model = OpticalNet(num_classes=num_classes).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    # Some training runs save the Linear under a different key shape; assume default.
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[ok] Loaded checkpoint '{path}' "
          f"(epoch {ckpt.get('epoch', '?')}, test_acc {ckpt.get('test_acc', '?')})")
    return model


# ============================================================================
# 2.  DATA LOADING
# ============================================================================
def load_breastmnist(npy_path=BREAST_IMAGES_NPY):
    """
    Return (images, labels) as torch tensors with images in [0,1] and shape
    (N, 224, 224).  Mirrors the convention of rogue_wave_analysis.py.
    """
    p = Path(npy_path)
    if p.exists():
        print(f"[ok] Loading images from {p}")
        imgs = np.load(p) / 255.0
        # rogue_wave_analysis.py only saves images, labels are loaded separately.
        # For SF-NOISE we DO need labels; try a matching labels file:
        labs_path = p.with_name('breast_labels.npy')
        if labs_path.exists():
            labs = np.load(labs_path).astype(np.int64).reshape(-1)
        else:
            print(f"[warn] '{labs_path}' not found.  "
                  "SF-NOISE will need labels from medmnist instead.")
            labs = None
        return (torch.from_numpy(imgs).float(),
                None if labs is None else torch.from_numpy(labs).long())

    if not USE_MEDMNIST_IF_NPY_MISSING:
        raise FileNotFoundError(p)

    print(f"[info] '{p}' not found, falling back to medmnist download...")
    import medmnist
    from medmnist import INFO
    from torchvision import transforms
    from skimage.color import rgb2gray
    info = INFO['breastmnist']
    DataClass = getattr(medmnist, info['python_class'])
    t = transforms.ToTensor()
    splits = []
    label_list = []
    for s in ('train', 'val', 'test'):
        ds = DataClass(split=s, transform=t, download=True, size=224)
        splits.append(np.array(ds.imgs))
        label_list.append(np.array(ds.labels))
    raw = np.concatenate(splits)        # (N, 224, 224, 3) RGB
    imgs = np.stack([rgb2gray(im) for im in raw])  # (N, 224, 224), float in [0,1]
    labs = np.concatenate(label_list).reshape(-1).astype(np.int64)
    return torch.from_numpy(imgs).float(), torch.from_numpy(labs).long()


def reproducible_test_split(imgs, labs, seed=42, test_frac=0.2):
    """Match the train/test split used in optical_net_train.py."""
    from sklearn.model_selection import train_test_split
    if labs is None:
        # cannot stratify without labels; return everything as "test"
        return imgs, None
    _, x_test, _, y_test = train_test_split(
        imgs.numpy(), labs.numpy(), test_size=test_frac, random_state=seed
    )
    return torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()


# ============================================================================
# 3.  SF-IO  --  Single-neuron I-O firing curve (Comment 1.1)
# ============================================================================
def figure_io_curve(model, test_images, n_alpha=60, alpha_min=0.01, alpha_max=1.5,
                    image_index=0):
    """
    Sweep the input-amplitude scalar alpha (which multiplies the data amplitude
    *before* modulation), holding everything else fixed.  Plot:

      (a) raw intensity at a representative caustic site vs alpha, with both
          fixed and adaptive thresholds overlaid, plus a sigmoid fit to the
          firing state under the fixed threshold;
      (b) total per-image spike count vs alpha under both thresholding
          conventions.

    The 'fixed threshold' is the adaptive threshold computed at alpha=1 and
    then held constant; this is the experimentally meaningful operating mode
    (calibrate once, then use the system).  The 'adaptive threshold' is the
    re-computed I_RW at each alpha, which by construction yields no firing
    change with alpha and is shown only to make the design choice explicit.
    """
    print("\n[SF-IO] Building I-O curve...")
    model.eval()
    img = test_images[image_index:image_index+1].to(DEVICE)  # (1,224,224)

    with torch.no_grad():
        # --- Calibration at alpha = 1 --------------------------------------
        _, dbg = model.propagate(img, amplitude_scale=1.0,
                                 return_intermediates=True)
        I_design = dbg['intensity'][0].cpu().numpy()           # (GRID,GRID)
        I_RW_design = dbg['threshold'][0].item()
        # Representative neuron = brightest caustic at design operating point.
        site_y, site_x = np.unravel_index(I_design.argmax(), I_design.shape)
        print(f"  representative caustic site = ({site_y},{site_x}); "
              f"I_RW_design = {I_RW_design:.3e}")

        # --- Sweep --------------------------------------------------------
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
        I_site_vs_alpha    = np.zeros(n_alpha)
        I_RW_adaptive      = np.zeros(n_alpha)
        spikes_fixed_thr   = np.zeros(n_alpha)
        spikes_adapt_thr   = np.zeros(n_alpha)

        # Fix the threshold to its alpha=1 value to make the I-O curve
        # physically meaningful (otherwise the threshold scales with the input).
        fixed_thr = torch.full((1, 1, 1), I_RW_design, device=DEVICE, dtype=DTYPE)

        for k, a in enumerate(tqdm(alphas, desc="  alpha sweep")):
            # fixed threshold
            _, dbgF = model.propagate(img, amplitude_scale=float(a),
                                      fixed_threshold=fixed_thr,
                                      return_intermediates=True)
            I = dbgF['intensity'][0].cpu().numpy()
            mask_fixed = (I > I_RW_design).astype(np.float32)
            # adaptive threshold (for completeness)
            _, dbgA = model.propagate(img, amplitude_scale=float(a),
                                      return_intermediates=True)
            mask_adapt = (dbgA['intensity'][0] > dbgA['threshold'][0]).cpu().numpy()

            I_site_vs_alpha[k] = I[site_y, site_x]
            I_RW_adaptive[k]   = dbgA['threshold'][0].item()
            spikes_fixed_thr[k] = mask_fixed.sum()
            spikes_adapt_thr[k] = mask_adapt.sum()

    # --- Plot --------------------------------------------------------------
    fig = plt.figure(figsize=(cm_to_in(17.5), cm_to_in(7.5)))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    # Panel (a) — local intensity vs alpha
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(alphas, I_site_vs_alpha, 'o-', ms=3, lw=1.0,
               color='#1F4E79', label=r'$I(x_0,y_0)$ at caustic site')
    ax1.axhline(I_RW_design, ls='--', lw=1.0, color='#C0392B',
                label=r'$I_{RW}$ (fixed, calibrated at $\alpha\!=\!1$)')
    ax1.loglog(alphas, I_RW_adaptive, ls=':', lw=1.0, color='#283747',
               label=r'$I_{RW}(\alpha)$ (adaptive)')
    ax1.set_xlabel(r'Input-amplitude scale $\alpha$', fontsize=7)
    ax1.set_ylabel(r'Local intensity (a.u.)', fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.legend(fontsize=6, frameon=False, loc='lower right')
    add_label(ax1, 'a)')

    # Panel (b) — total spike count vs alpha
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(alphas, spikes_fixed_thr, 'o-', ms=3, lw=1.0,
                 color='#1F4E79', label='Fixed threshold')
    ax2.semilogx(alphas, spikes_adapt_thr, 's--', ms=3, lw=1.0,
                 color='#7F8C8D', label='Adaptive threshold')
    # Sigmoid fit to fixed-threshold curve in log-alpha
    try:
        from scipy.optimize import curve_fit
        def sigmoid(x, L, x0, k, b): return L / (1 + np.exp(-k * (x - x0))) + b
        log_a = np.log10(alphas)
        sat = spikes_fixed_thr.max()
        p0 = [sat, np.log10(np.sqrt(I_RW_design / I_site_vs_alpha.max())), 5.0, 0.0]
        popt, _ = curve_fit(sigmoid, log_a, spikes_fixed_thr, p0=p0, maxfev=20000)
        a_dense = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 500)
        ax2.semilogx(a_dense, sigmoid(np.log10(a_dense), *popt), '-',
                     lw=1.0, color='#C0392B', alpha=0.8,
                     label='Sigmoid fit (fixed)')
    except Exception as e:
        print(f"  [warn] sigmoid fit failed: {e}")
    ax2.set_xlabel(r'Input-amplitude scale $\alpha$', fontsize=7)
    ax2.set_ylabel(r'Total spike count per inference', fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.legend(fontsize=6, frameon=False, loc='upper left')
    add_label(ax2, 'b)')

    fig.suptitle("Single-neuron I\u2013O firing curve", fontsize=8, y=1.02)
    save_fig(fig, "SF_IO_firing_curve")
    plt.close(fig)


# ============================================================================
# 4.  SF-REPEAT  --  Repeatability under detector noise (Comment 2.2)
# ============================================================================
def add_camera_noise(intensity, full_well=10500.0, read_noise_e=2.5,
                     gain_e_per_dn=40.0, bit_depth=8, scene_peak_dn=200.0):
    """
    Simulate a single noisy capture by a global-shutter CMOS camera
    (parameters chosen to bracket the FLIR BFS-U3-51S5M / Sony IMX250:
    full well ~10.5 ke-, read noise ~2.5 e-, 8-bit readout used by the
    experimental pipeline).

    Mapping:  intensity  ->  electrons  ->  Poisson  +  Gaussian read noise
              -> 8-bit quantization -> back to float intensity.
    """
    I = intensity.detach().cpu().numpy()
    # Normalize the scene so the brightest caustic sits at scene_peak_dn out
    # of 2**bit_depth-1 counts (mimics the gain knob the user sets on the
    # bench).  Then convert DN -> electrons.
    dn = I / I.max() * scene_peak_dn
    e_photo = np.clip(dn * gain_e_per_dn, 0, full_well)
    # Poisson shot noise
    e_with_shot = np.random.poisson(e_photo).astype(np.float32)
    # Gaussian read noise
    e_with_read = e_with_shot + np.random.normal(0, read_noise_e, e_photo.shape)
    # Quantize back to DN, clip to bit depth, then return as float intensity
    dn_meas = np.clip(np.round(e_with_read / gain_e_per_dn), 0, 2**bit_depth - 1)
    I_meas  = dn_meas / scene_peak_dn * I.max()
    return torch.from_numpy(I_meas).to(intensity.device).to(intensity.dtype)


def figure_repeatability(model, test_images, image_index=0, n_realizations=100):
    """
    100 noisy reads of the same input + mask.  Apply the rogue-wave threshold
    to each noisy intensity, build the binary spike map, then compute pairwise
    Jaccard index across all C(N,2) realizations.

    Reports mean +- std and plots a histogram.
    """
    print("\n[SF-REPEAT] Repeatability under detector noise...")
    model.eval()
    img = test_images[image_index:image_index+1].to(DEVICE)

    with torch.no_grad():
        _, dbg = model.propagate(img, return_intermediates=True)
        I_clean = dbg['intensity'][0]   # (GRID,GRID), on device

        masks = np.zeros((n_realizations, GRID, GRID), dtype=bool)
        for k in tqdm(range(n_realizations), desc='  noisy reads'):
            I_noisy = add_camera_noise(I_clean.unsqueeze(0)).squeeze(0)
            # adaptive threshold on the noisy intensity (same as in inference)
            soft_mask, thr = model.rogue_threshold(I_noisy.unsqueeze(0))
            masks[k] = (I_noisy.cpu().numpy() > thr.item())

    # Pairwise Jaccard
    print("  computing pairwise Jaccard...")
    flat = masks.reshape(n_realizations, -1)
    sums = flat.sum(axis=1)
    n_pairs = n_realizations * (n_realizations - 1) // 2
    jaccards = np.empty(n_pairs, dtype=np.float32)
    p = 0
    for i in tqdm(range(n_realizations), desc='  pairs'):
        for j in range(i+1, n_realizations):
            inter = np.logical_and(flat[i], flat[j]).sum()
            union = sums[i] + sums[j] - inter
            jaccards[p] = inter / max(1, union)
            p += 1

    j_mean = jaccards.mean()
    j_std  = jaccards.std()
    print(f"  mean Jaccard = {j_mean:.3f} +- {j_std:.3f}")

    # --- Plot --------------------------------------------------------------
    fig = plt.figure(figsize=(cm_to_in(8.5), cm_to_in(6.5)))
    ax = fig.add_subplot(111)
    ax.hist(jaccards, bins=40, color='#5DADE2', edgecolor='#2E86C1', linewidth=0.4)
    ax.axvline(j_mean, color='#C0392B', ls='--', lw=1.0,
               label=fr'mean = {j_mean:.3f} $\pm$ {j_std:.3f}')
    ax.set_xlabel('Pairwise Jaccard index of binary spike maps', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    ax.tick_params(labelsize=6)
    fig.suptitle("Spike-map repeatability "
                 f"({n_realizations} noisy realizations)", fontsize=8, y=1.02)
    save_fig(fig, "SF_REPEAT_jaccard")
    plt.close(fig)
    return j_mean, j_std


# ============================================================================
# 5.  SF-NOISE  --  Input-noise robustness (Comment 2.2)
# ============================================================================
def figure_noise_robustness(model, test_images, test_labels,
                            snr_db_list=(30, 25, 20, 15, 10),
                            n_trials_per_snr=3):
    """
    Add AWGN to the *input amplitude* (matching the BreastMNIST [0,1] range)
    at several SNR levels and report classification accuracy.  Repeats each
    SNR n_trials_per_snr times with different noise realizations to get an
    error bar.

    SNR convention:  SNR_dB = 10 log10( var(signal) / var(noise) ),
    measured per image on its non-zero support.
    """
    print("\n[SF-NOISE] Input-noise robustness sweep...")
    if test_labels is None:
        raise RuntimeError("SF-NOISE needs labels; please ensure breast_labels.npy "
                           "is present or that medmnist is reachable.")
    model.eval()
    test_images = test_images.to(DEVICE)
    test_labels = test_labels.to(DEVICE).reshape(-1)

    # First measure clean accuracy as a sanity baseline
    with torch.no_grad():
        logits = []
        for i in range(0, len(test_images), 32):
            logits.append(model(test_images[i:i+32]).cpu())
        clean_acc = (torch.cat(logits).argmax(1) == test_labels.cpu()).float().mean().item()
    print(f"  clean test accuracy = {clean_acc*100:.2f}%")

    snr_db_list = list(snr_db_list)
    acc_mean = np.zeros(len(snr_db_list))
    acc_std  = np.zeros(len(snr_db_list))

    for s_idx, snr_db in enumerate(snr_db_list):
        accs = []
        for t in range(n_trials_per_snr):
            noisy = test_images.clone()
            # per-image SNR-calibrated AWGN on the input amplitude
            sig_pow = (test_images ** 2).mean(dim=(1, 2), keepdim=True) + 1e-12
            noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(test_images) * torch.sqrt(noise_pow)
            noisy = (noisy + noise).clamp(min=0.0)  # amplitude stays non-negative
            with torch.no_grad():
                preds = []
                for i in range(0, len(noisy), 32):
                    preds.append(model(noisy[i:i+32]).argmax(1).cpu())
                preds = torch.cat(preds)
            accs.append((preds == test_labels.cpu()).float().mean().item())
        acc_mean[s_idx] = np.mean(accs)
        acc_std[s_idx]  = np.std(accs)
        print(f"  SNR = {snr_db:>3} dB:  acc = {acc_mean[s_idx]*100:.2f} "
              f"+- {acc_std[s_idx]*100:.2f}")

    # --- Plot --------------------------------------------------------------
    fig = plt.figure(figsize=(cm_to_in(8.5), cm_to_in(6.5)))
    ax = fig.add_subplot(111)
    ax.errorbar(snr_db_list, acc_mean * 100, yerr=acc_std * 100,
                fmt='o-', color='#1F4E79', ecolor='#7F8C8D',
                ms=4, lw=1.0, capsize=2.5)
    ax.axhline(clean_acc * 100, ls='--', lw=1.0, color='#27AE60',
               label=f'Clean baseline ({clean_acc*100:.1f}%)')
    ax.set_xlabel('Input-amplitude SNR (dB)', fontsize=7)
    ax.set_ylabel('Test accuracy (%)', fontsize=7)
    ax.invert_xaxis()  # higher SNR on the left, degrading toward right
    ax.legend(fontsize=6, frameon=False, loc='lower right')
    ax.tick_params(labelsize=6)
    fig.suptitle("Classification accuracy vs. input-amplitude SNR",
                 fontsize=8, y=1.02)
    save_fig(fig, "SF_NOISE_snr_sweep")
    plt.close(fig)


# ============================================================================
# 6.  SF-DIST  --  Propagation-distance sweep (Comment 2.1)
#
# We deliberately use a *random* (untrained) phase mask here, because the
# trained mask is optimized for z = 40 cm and degrades unfairly off-design.
# This isolates the physics the reviewer is asking about: "do rogue waves
# form at shorter z?"  If you also want the trained-mask accuracy curve,
# enable the include_trained_accuracy flag (slower, requires retraining
# the classifier at each z for a fair comparison; we skip that here).
# ============================================================================
def figure_distance_sweep(test_images,
                          z_list_cm=(10, 20, 30, 40, 50),
                          n_samples=30, seed=42):
    """
    For each propagation distance, run n_samples random-image / random-phase
    realizations and report (mean +- std) of:
      - number of spikes per inference (under I_RW = 2 I_sig criterion)
      - I_max / I_sig ratio
    """
    print("\n[SF-DIST] Propagation-distance sweep...")
    rng = np.random.default_rng(seed)
    # pick n_samples random images
    idx = rng.choice(len(test_images), size=min(n_samples, len(test_images)),
                     replace=False)
    imgs = test_images[idx].to(DEVICE)  # (N,224,224)

    # build a random phase mask once, replicate at each z
    torch.manual_seed(seed)
    rand_mask = PhasePattern().to(DEVICE)
    phi = rand_mask().detach()  # (224,224)
    phi_pad = F.pad(phi.unsqueeze(0).unsqueeze(0),
                    ((GRID-INPUT_SIZE)//2,)*4, mode='constant', value=0)
    src = _build_source_field().to(DEVICE)

    z_list_m = np.array(z_list_cm) * 1e-2
    n_spikes_mean = np.zeros_like(z_list_m)
    n_spikes_std  = np.zeros_like(z_list_m)
    ratio_mean    = np.zeros_like(z_list_m)
    ratio_std     = np.zeros_like(z_list_m)

    for zi, z in enumerate(z_list_m):
        n_spikes = []
        ratios   = []
        for k in tqdm(range(len(imgs)), desc=f'  z = {z*100:.0f} cm'):
            img = imgs[k:k+1]
            img_pad = F.pad(img.unsqueeze(1),
                            ((GRID-INPUT_SIZE)//2,)*4, mode='constant', value=0)
            modulated = src * img_pad * torch.exp(1j * phi_pad)
            U = utils.propagation_ASM(modulated, RESOLUTION, WAVELENGTH, float(z))
            I = (torch.abs(U) ** 2).squeeze().cpu().numpy()
            flat = I.flatten()
            tops = np.sort(flat)[int(len(flat) * 2 / 3):]
            I_sig = tops.mean()
            I_RW  = 2 * I_sig
            n_spikes.append(int((I > I_RW).sum()))
            ratios.append(I.max() / max(I_sig, 1e-20))
        n_spikes_mean[zi] = np.mean(n_spikes)
        n_spikes_std[zi]  = np.std(n_spikes)
        ratio_mean[zi]    = np.mean(ratios)
        ratio_std[zi]     = np.std(ratios)
        print(f"    spikes = {n_spikes_mean[zi]:.1f} +- {n_spikes_std[zi]:.1f},"
              f"  I_max/I_sig = {ratio_mean[zi]:.2f} +- {ratio_std[zi]:.2f}")

    # --- Plot --------------------------------------------------------------
    fig = plt.figure(figsize=(cm_to_in(17.5), cm_to_in(7.0)))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(z_list_cm, n_spikes_mean, yerr=n_spikes_std,
                 fmt='o-', color='#1F4E79', ecolor='#7F8C8D',
                 ms=4, lw=1.0, capsize=2.5)
    ax1.set_xlabel('Propagation distance $z$ (cm)', fontsize=7)
    ax1.set_ylabel('Number of spikes per inference', fontsize=7)
    ax1.tick_params(labelsize=6)
    add_label(ax1, 'a)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(z_list_cm, ratio_mean, yerr=ratio_std,
                 fmt='s-', color='#C0392B', ecolor='#7F8C8D',
                 ms=4, lw=1.0, capsize=2.5)
    ax2.axhline(2.0, ls=':', lw=0.8, color='#283747',
                label=r'$I_{RW}/I_{sig} = 2$ (rogue criterion)')
    ax2.set_xlabel('Propagation distance $z$ (cm)', fontsize=7)
    ax2.set_ylabel(r'$I_{\max}/I_{sig}$', fontsize=7)
    ax2.legend(fontsize=6, frameon=False, loc='lower right')
    ax2.tick_params(labelsize=6)
    add_label(ax2, 'b)')

    fig.suptitle("Rogue-wave statistics vs. propagation distance "
                 "(random phase mask)", fontsize=8, y=1.02)
    save_fig(fig, "SF_DIST_distance_sweep")
    plt.close(fig)


# ============================================================================
# 7.  SF-CASC  --  Two-layer cascade demo (Comment 2.6)
#
# We define a 2-layer rogue-wave network and (optionally) fine-tune it for a
# small number of epochs on BreastMNIST.  The purpose of this figure is *not*
# to set a new accuracy record but to demonstrate that the differentiable
# digital twin supports cascaded propagation-threshold stages.
#
# To keep this fast, we re-use the input encoding from the single-layer model
# and re-image the soft-mask output of layer 1 as the amplitude going into
# layer 2 (preserving the (480,480) grid).  Layer-2 phase mask has its own
# trainable parameters.
# ============================================================================
class TwoLayerOpticalNet(nn.Module):
    def __init__(self, num_classes=2, steepness=100.0):
        super().__init__()
        self.resolution = RESOLUTION
        self.wavelength = WAVELENGTH

        self.register_buffer('source_field', _build_source_field())
        self.phase_1 = PhasePattern(INPUT_SIZE, INPUT_SIZE, SUPERPIXEL)
        # Layer-2 phase mask spans the full propagation grid (already 480x480).
        # We keep the same superpixel size for parameter parity.
        self.phase_2_vals = nn.Parameter(
            torch.rand(GRID // SUPERPIXEL, GRID // SUPERPIXEL) * 2 * math.pi
        )
        self.rogue = RogueWaveThreshold(steepness=steepness)
        self.pool  = nn.AvgPool2d(POOL, POOL)
        feat = (GRID // POOL) ** 2
        self.classifier = nn.Linear(feat, num_classes)

        self.pad_h = self.pad_w = (GRID - INPUT_SIZE) // 2

    def _pad(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
                     mode='constant', value=0)

    def _phase_2(self):
        p = torch.repeat_interleave(self.phase_2_vals, SUPERPIXEL, dim=0)
        p = torch.repeat_interleave(p, SUPERPIXEL, dim=1)
        return p[:GRID, :GRID]

    def forward(self, input_amp):
        # ---- LAYER 1 ----
        x = self._pad(input_amp)
        phi1 = self._pad(self.phase_1())
        U1 = utils.propagation_ASM(
            self.source_field * x * torch.exp(1j * phi1),
            self.resolution, self.wavelength, PROP_DISTANCE
        )
        I1 = (torch.abs(U1) ** 2).squeeze(1)
        soft_mask_1, _ = self.rogue(I1)              # (B,GRID,GRID)
        # ---- LAYER 2 ----
        # Re-modulate: soft_mask_1 plays the role of amplitude, phase_2 is the mask
        amp2 = soft_mask_1.unsqueeze(1)              # (B,1,GRID,GRID)
        phi2 = self._phase_2().unsqueeze(0).unsqueeze(0)
        U2 = utils.propagation_ASM(
            amp2 * torch.exp(1j * phi2),
            self.resolution, self.wavelength, PROP_DISTANCE
        )
        I2 = (torch.abs(U2) ** 2).squeeze(1)
        soft_mask_2, _ = self.rogue(I2)
        # ---- READOUT ----
        pooled = self.pool(soft_mask_2.unsqueeze(1)).flatten(1)
        return self.classifier(pooled), {'I1': I1, 'mask1': soft_mask_1,
                                         'I2': I2, 'mask2': soft_mask_2}


def figure_cascade_demo(test_images, test_labels,
                        train_images=None, train_labels=None,
                        n_epochs=200, batch_size=20, lr=1e-4):
    """
    Trains a 2-layer cascade for a short run and produces:
      (a) loss/accuracy curves vs. epoch,
      (b) example layer-1 and layer-2 intensities and spike masks for one
          test image.

    If train_images / train_labels are None, the function reads the same
    split as the manuscript and uses 80% as training (matches the
    train_test_split(seed=42, test_size=0.2) in optical_net_train.py).
    """
    print("\n[SF-CASC] Two-layer cascade demonstration...")
    if test_labels is None:
        raise RuntimeError("SF-CASC needs labels; please ensure breast_labels.npy "
                           "is present or that medmnist is reachable.")

    # Build dataloaders that match the manuscript split
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    if train_images is None:
        # We assume the caller already passed (full_images, full_labels)
        # via load_breastmnist() and a separate reproducible split.
        raise RuntimeError("Pass training images/labels explicitly.")

    train_ds = TensorDataset(train_images.float(), train_labels.long())
    test_ds  = TensorDataset(test_images.float(),  test_labels.long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = TwoLayerOpticalNet().to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = [], [], [], []
    for ep in range(n_epochs):
        # ---- train ----
        model.train()
        ep_loss, n_correct, n_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE).reshape(-1)
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            ep_loss   += loss.item()
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total   += y.size(0)
        train_loss_hist.append(ep_loss / len(train_dl))
        train_acc_hist.append(100 * n_correct / n_total)
        # ---- eval ----
        model.eval()
        ep_loss, n_correct, n_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE).reshape(-1)
                logits, _ = model(x)
                ep_loss   += crit(logits, y).item()
                n_correct += (logits.argmax(1) == y).sum().item()
                n_total   += y.size(0)
        test_loss_hist.append(ep_loss / len(test_dl))
        test_acc_hist.append(100 * n_correct / n_total)
        print(f"  ep {ep+1:3d}/{n_epochs}: "
              f"train acc {train_acc_hist[-1]:5.2f}%, "
              f"test acc {test_acc_hist[-1]:5.2f}%")

    # --- Forward one sample to show layer-by-layer intermediates -----------
    model.eval()
    with torch.no_grad():
        x0 = test_images[0:1].to(DEVICE)
        _, dbg = model(x0)

    fig = plt.figure(figsize=(cm_to_in(17.5), cm_to_in(10.0)))
    gs  = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)

    # Loss / Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_loss_hist, '-', color='#1F4E79', lw=1.0, label='train')
    ax1.plot(test_loss_hist,  '-', color='#C0392B', lw=1.0, label='test')
    ax1.set_xlabel('Epoch', fontsize=7); ax1.set_ylabel('Loss', fontsize=7)
    ax1.tick_params(labelsize=6); ax1.legend(fontsize=6, frameon=False)
    add_label(ax1, 'a)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train_acc_hist, '-', color='#1F4E79', lw=1.0, label='train')
    ax2.plot(test_acc_hist,  '-', color='#C0392B', lw=1.0, label='test')
    ax2.set_xlabel('Epoch', fontsize=7); ax2.set_ylabel('Accuracy (%)', fontsize=7)
    ax2.tick_params(labelsize=6); ax2.legend(fontsize=6, frameon=False)
    add_label(ax2, 'b)')

    # Layer-1 intensity
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(dbg['I1'][0].cpu().numpy(), cmap='magma')
    ax3.set_title('Layer-1 intensity', fontsize=7); ax3.set_xticks([]); ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
    add_label(ax3, 'c)')

    # Layer-1 spikes
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(dbg['mask1'][0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Layer-1 spike map', fontsize=7); ax4.set_xticks([]); ax4.set_yticks([])
    add_label(ax4, 'd)')

    # Layer-2 intensity
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(dbg['I2'][0].cpu().numpy(), cmap='magma')
    ax5.set_title('Layer-2 intensity', fontsize=7); ax5.set_xticks([]); ax5.set_yticks([])
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
    add_label(ax5, 'e)')

    # Layer-2 spikes
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(dbg['mask2'][0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax6.set_title('Layer-2 spike map', fontsize=7); ax6.set_xticks([]); ax6.set_yticks([])
    add_label(ax6, 'f)')

    fig.suptitle("Two-layer rogue-wave cascade trained end-to-end",
                 fontsize=8, y=1.00)
    save_fig(fig, "SF_CASC_cascade")
    plt.close(fig)


# ============================================================================
# 7b. TRAIN SINGLE-LAYER MODEL FROM SCRATCH  (used when no checkpoint available)
#
# Mirrors the recipe in optical_net_train.py:
#   Adam(lr=1e-4) + CosineAnnealingLR + CrossEntropyLoss
#   batch_size=20, early stopping on test loss with patience=50, up to 400 ep.
#
# Returns the best model (by test loss) ready for the figure functions.
# ============================================================================
def train_single_layer_from_scratch(train_images, train_labels,
                                    test_images, test_labels,
                                    n_epochs=SINGLE_LAYER_EPOCHS,
                                    batch_size=SINGLE_LAYER_BATCH,
                                    lr=SINGLE_LAYER_LR,
                                    patience=SINGLE_LAYER_PATIENCE,
                                    save_path='best_model_freshly_trained'):
    print(f"\n[train] Training single-layer OpticalNet from scratch "
          f"(<= {n_epochs} epochs, early stop patience {patience})...")
    if train_labels is None or test_labels is None:
        raise RuntimeError("Single-layer training needs labels; make sure "
                           "breast_labels.npy is present or medmnist is reachable.")

    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(train_images.float(), train_labels.long())
    test_ds  = TensorDataset(test_images.float(),  test_labels.long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = OpticalNet(num_classes=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params}")

    best_test_loss = float('inf')
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = {'train_loss': [], 'test_loss': [],
               'train_acc':  [], 'test_acc':  []}

    for ep in range(n_epochs):
        # ---- train ----
        model.train()
        s_loss, s_correct, s_total = 0.0, 0, 0
        for x, y in train_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE).reshape(-1)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            s_loss    += loss.item()
            s_correct += (logits.argmax(1) == y).sum().item()
            s_total   += y.size(0)
        train_loss = s_loss / len(train_dl)
        train_acc  = 100 * s_correct / s_total

        # ---- eval ----
        model.eval()
        s_loss, s_correct, s_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(DEVICE)
                y = y.to(DEVICE).reshape(-1)
                logits = model(x)
                s_loss    += crit(logits, y).item()
                s_correct += (logits.argmax(1) == y).sum().item()
                s_total   += y.size(0)
        test_loss = s_loss / len(test_dl)
        test_acc  = 100 * s_correct / s_total

        sched.step()
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  ep {ep+1:3d}/{n_epochs}: "
              f"train {train_loss:.4f}/{train_acc:5.2f}%  "
              f"test {test_loss:.4f}/{test_acc:5.2f}%  "
              f"lr {opt.param_groups[0]['lr']:.2e}"
              + ("  *" if patience_counter == 0 else ""))

        if patience_counter >= patience:
            print(f"  early stopping at epoch {ep+1}")
            break

    # restore best weights
    model.load_state_dict(best_state)
    model.eval()
    print(f"[train] Done. Best test loss {best_test_loss:.4f} at epoch {best_epoch+1}, "
          f"test acc {history['test_acc'][best_epoch]:.2f}%")

    # save checkpoint so a re-run can skip training
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_state,
        'test_loss': best_test_loss,
        'test_acc':  history['test_acc'][best_epoch],
        'history':   history,
    }, save_path)
    print(f"[train] Saved fresh checkpoint to '{save_path}'")
    return model


# ============================================================================
# 8.  MAIN
# ============================================================================
def main():
    # ---- Load data -----------------------------------------------------------
    all_images, all_labels = load_breastmnist()

    # Reproduce the train_test_split used in optical_net_train.py exactly,
    # so the test set we evaluate on is the same as the manuscript's.
    from sklearn.model_selection import train_test_split
    if all_labels is None:
        raise RuntimeError("Labels are required: breast_labels.npy missing AND "
                           "medmnist unavailable. Cannot proceed.")
    x_train, x_test, y_train, y_test = train_test_split(
        all_images.numpy(), all_labels.numpy(),
        test_size=0.2, random_state=42
    )
    train_images = torch.from_numpy(x_train).float()
    train_labels = torch.from_numpy(y_train).long()
    test_images  = torch.from_numpy(x_test).float()
    test_labels  = torch.from_numpy(y_test).long()
    print(f"[ok] split: {len(train_images)} train / {len(test_images)} test")

    # ---- Obtain a trained single-layer model --------------------------------
    if TRAIN_FROM_SCRATCH:
        model = train_single_layer_from_scratch(
            train_images, train_labels, test_images, test_labels,
            n_epochs=SINGLE_LAYER_EPOCHS,
            batch_size=SINGLE_LAYER_BATCH,
            lr=SINGLE_LAYER_LR,
            patience=SINGLE_LAYER_PATIENCE,
        )
    else:
        model = load_trained_model(CHECKPOINT_PATH, num_classes=2)

    # ---- Figures -------------------------------------------------------------
    # Each block is independent; comment out anything you have already produced.

    figure_io_curve(model, test_images, image_index=0)
    figure_repeatability(model, test_images, image_index=0, n_realizations=100)
    figure_noise_robustness(model, test_images, test_labels,
                            snr_db_list=(30, 25, 20, 15, 10, 5, 3, 1),
                            n_trials_per_snr=3)
    figure_distance_sweep(test_images,
                          z_list_cm=(1,5,10, 20, 30, 40),
                          n_samples=30)
    """
    # The two-layer cascade demo trains its own model from scratch.
    figure_cascade_demo(
        test_images=test_images, test_labels=test_labels,
        train_images=train_images, train_labels=train_labels,
        n_epochs=200,           # bump to 50-80 if you want better cascade accuracy
        batch_size=20, lr=1e-4
    )"""

    print("\nDone.  All figures saved to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()