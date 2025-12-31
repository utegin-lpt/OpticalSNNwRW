import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utils
import math
from scipy.stats import rayleigh
import scienceplots  # pip install SciencePlots
from tqdm import tqdm
from matplotlib import cm

# =============================================================================
# 1. BASIC FUNCTIONS (SETUP & UTILS)
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_input(x, target_size, input_size):
    """Pads the input to the target grid size."""
    pad_h = (target_size[0] - input_size[0]) // 2
    pad_w = (target_size[1] - input_size[1]) // 2
    if x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        
    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    return x_padded.squeeze()

def create_checkerboard(dimension, superpixel, seed=42):
    """Generates a random checkerboard pattern (between 0-1)."""
    torch.manual_seed(seed)
    height, width = dimension
    
    grid_h = int(np.ceil(height / superpixel))
    grid_w = int(np.ceil(width / superpixel))
    
    random_grid = torch.rand(grid_h, grid_w)
    
    pattern = torch.repeat_interleave(random_grid, superpixel, dim=0)
    pattern = torch.repeat_interleave(pattern, superpixel, dim=1)
    
    pattern = pattern[:height, :width]
    return pattern

def run_simulation_batch(images, superpixel, mode='amp_info', seed=42):
    """
    Runs the simulation and returns I_max / I_sig (I_1/3) ratios.
    """
    # Physical Parameters
    lambda_c = 632e-9
    xres, yres = 8e-6, 8e-6
    grid_size_y, grid_size_x = 400, 400
    data_size = images.shape[1:]
    
    spacewidth = grid_size_x * xres
    spacelength = grid_size_y * yres
    x = torch.linspace(-spacewidth*0.5, spacewidth*0.5, int(spacewidth/xres))
    y = torch.linspace(-spacelength*0.5, spacelength*0.5, int(spacelength/yres))
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    Pavg = 0.39
    beam_width_x = 224 * xres
    beam_width_y = 224 * yres
    x_fwhm, y_fwhm = beam_width_x / 2.355, beam_width_y / 2.355
    A_gaussian = torch.sqrt((Pavg / (math.pi * x_fwhm * y_fwhm)) * torch.exp(-((X**2) / (2 * beam_width_x**2) + (Y**2) / (2 * beam_width_y**2))))

    random_pattern = create_checkerboard((224, 224), superpixel, seed=seed)
    random_pattern_padded = pad_input(random_pattern, (grid_size_y, grid_size_x), data_size)

    ratios = []
    output_intensities = []

    print(f"Simulation Starting: Mode={mode}, Superpixel={superpixel}")
    
    for i in tqdm(range(len(images))):
        img = images[i].float()
        
        # --- NORMALIZATION (Critical Step) ---
        # Regardless of the input range (0-510, 0-255, etc.), 
        # it is compressed to 0-1 here.
        if img.max() > img.min(): # Prevent division by zero error
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = torch.zeros_like(img)

        img_padded = pad_input(img, (grid_size_y, grid_size_x), data_size)
        
        if mode == 'amp_info':
            # Information in Amplitude: [0,1]
            A_input = A_gaussian * img_padded
            # Random Phase: [0, 2pi]
            phi_input = 2 * math.pi * random_pattern_padded
            
        elif mode == 'phase_info':
            # Information in Phase: [0, 2pi] (img is already 0-1, multiply by 2pi)
            phi_input = 2 * math.pi * img_padded 
            # Random Amplitude: [0, 1] x Gaussian Profile
            A_input = A_gaussian * random_pattern_padded

        U_in = A_input * torch.exp(1j * phi_input)
        
        U_out = utils.propagation_ASM(U_in.view(1, 1, grid_size_y, grid_size_x), 
                                      [xres, yres], lambda_c, 40e-2).view(grid_size_y, grid_size_x)
        
        I_out = np.square(np.abs(U_out.detach().cpu().numpy()))
        
        # --- STATISTICS CALCULATION (I_max / I_1/3) ---
        flat_I = I_out.flatten()
        mx = np.max(flat_I)
        
        # Top 1/3 significant wave height
        tops = np.sort(flat_I)[int(len(flat_I>1e-6) * (2/3)):] 
        I_sig = np.mean(tops)
        
        # Ratio calculation: I_max / I_sig
        if I_sig > 0:
            ratio = mx / I_sig
        else:
            ratio = 0
        
        ratios.append(ratio)
        output_intensities.append(I_out)

    return np.array(ratios), np.array(output_intensities), A_input, phi_input

def cm_to_inch(value):
    return value / 2.54

# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def add_subplot_label(ax, label):
    """Adds (a), (b), etc., labels to the top-left corner of the plot."""
    # Coordinates x=-0.05, y=1.05 fall slightly outside axes, top-left
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, 
            fontsize=6, fontweight='bold', va='top', ha='right')

def generate_scientific_plots(I_out, A_in, phi_in, filename_prefix="Analyze"):
    """
    Generates 2D plots and an aesthetic histogram with normalized data.
    Subplot labels (a, b, c...) are added.
    """
    # 1. Data Normalization [0, 1]
    I_norm = (I_out - I_out.min()) / (I_out.max() - I_out.min())
    
    # 2. Recalculate Statistics on Normalized Data
    flat_data = I_norm.flatten()
    tops = np.sort(flat_data)[int(len(flat_data)*(2/3)):]
    I_sig_norm = np.mean(tops)
    I_rw_thresh_norm = 2 * I_sig_norm
    
    # Mask
    binary_mask = (I_norm > I_rw_thresh_norm).astype(float)

    # Phase Adjustment: [-pi, pi] -> [0, 2pi]
    phi_disp = (np.angle(np.exp(1j * phi_in.cpu().numpy())) + 2*np.pi) % (2*np.pi)

    # Plot Settings
    try:
        plt.style.use(['science', 'nature'])
    except:
        plt.style.use('default')

    fig = plt.figure(figsize=(cm_to_inch(17.5), cm_to_inch(10)))
    
    # 2 Rows, 3 Columns Grid
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.35)

    # --- ROW 1 ---
    
    # Col 1: Amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.abs(A_in.cpu().numpy())/np.max(np.abs(A_in.cpu().numpy())), cmap='gray')
    ax1.set_title('Amplitude (Input)', fontsize=7)
    ax1.set_xticks([])
    ax1.set_yticks([])
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Amplitude (a.u.)', fontsize=5)
    cbar1.ax.tick_params(labelsize=5)
    add_subplot_label(ax1, 'a)')

    # Col 2: Phase [0 - 2pi]
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(phi_disp, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax2.set_title(r'Phase (Input)', fontsize=7)
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Phase (rad)', fontsize=5)
    cbar2.ax.tick_params(labelsize=5)
    add_subplot_label(ax2, 'b)')

    # Col 3: Recorded Pattern (2D, Normalized)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(I_norm, cmap='magma', vmin=0, vmax=1)
    ax3.set_title('Output Pattern', fontsize=7)
    ax3.set_xticks([])
    ax3.set_yticks([])
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r'Normalized Intensity (a.u.)', fontsize=5)
    cbar3.ax.tick_params(labelsize=5)
    add_subplot_label(ax3, 'c)')

    # --- ROW 2 ---

    # Col 1 & 2: Histogram (Merged)
    ax4 = fig.add_subplot(gs[1, 0:2])
    
    # Desired Pastel Blue Color (Peacock Blue/Cornflower)
    pastel_blue = '#5DADE2' 
    
    counts, bins, _ = ax4.hist(flat_data, bins='auto', density=True, stacked = True, 
                                color=pastel_blue, label='Pixel Dist.')
    
    # Lines
    ax4.axvline(I_rw_thresh_norm, color='#E74C3C', linestyle='--', linewidth=1.5, label=r'$I_{RW} (2 \cdot I_{1/3})$')
    ax4.axvline(I_sig_norm, color='#283747', linestyle='-.', linewidth=1.5, label=r'$I_{sig} (I_{1/3})$')
    
    # Rayleigh Fit
    params = rayleigh.fit(flat_data)
    x_ax = np.linspace(0, 1, 200)
    ax4.plot(x_ax, rayleigh.pdf(x_ax, *params), 'k-', linewidth=0.8, alpha=0.7, label='Rayleigh Fit')

    ax4.set_title('Intensity Histogram', fontsize=7)
    ax4.set_xlabel('Normalized Intensity (a.u.)', fontsize=6)
    ax4.set_ylabel('Probability Density', fontsize=6)
    ax4.legend(fontsize=6, loc='upper right', frameon=False)
    ax4.tick_params(labelsize=5)
    ax4.set_xlim(0, 1)
    add_subplot_label(ax4, 'd)')

    # Col 3: Binary Mask
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(r'Rogue Waves ($I > I_{RW}$)', fontsize=7)
    ax5.set_xticks([])
    ax5.set_yticks([])
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar5.ax.set_yticklabels(['Normal', 'Rogue'], fontsize=5)
    add_subplot_label(ax5, 'e)')

    plt.savefig(f"{filename_prefix}_CombinedPlot.png", dpi=1200, bbox_inches='tight')
    plt.savefig(f"{filename_prefix}_CombinedPlot.svg",format="svg", dpi=1200, bbox_inches='tight')
    plt.show()

def plot_qq(data, filename):
    """Q-Q Plot according to Rayleigh distribution."""
    try:
        plt.style.use(['science', 'nature'])
    except:
        pass
        
    plt.figure(figsize=(cm_to_inch(8), cm_to_inch(8)))
    
    data_sorted = np.sort(data.flatten())
    
    loc, scale = rayleigh.fit(data_sorted)
    
    probs = np.linspace(0.01, 0.99, len(data_sorted))
    theoretical_quantiles = rayleigh.ppf(probs, loc=loc, scale=scale)
    
    step = max(1, len(data_sorted) // 5000)
    
    # Match Q-Q plot color as well
    plt.scatter(theoretical_quantiles[::step], data_sorted[::step], 
                s=2, alpha=0.5, color='#3498DB', label='Sample')
    
    max_val = max(theoretical_quantiles.max(), data_sorted.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Reference')
    
    plt.title('Q-Q Plot (Rayleigh)', fontsize=8)
    plt.xlabel('Theoretical Quantiles', fontsize=7)
    plt.ylabel('Sample Quantiles', fontsize=7)
    plt.legend(fontsize=6)
    plt.grid(True, alpha=0.2)
    plt.savefig(filename, dpi=300)
    plt.show()

# =============================================================================
# 3. MAIN WORKFLOW
# =============================================================================

def main():
    try:
        df = torch.Tensor(np.load("breast_images.npy")/255).to(torch.float16)
    except FileNotFoundError:
        print("ERROR: 'breast_images.npy' not found.")
        return

    # --- ANALYSIS 1: Superpixel=8, Info=Amp, Random=Phase ---
    print("\n--- ANALYSIS 1: Superpixel=8, Info=Amp, Random=Phase ---")
    
    sp_size = 8
    ratios, intensities, _, _ = run_simulation_batch(df, sp_size, mode='amp_info', seed=42)
    
    min_idx = np.argmin(ratios)
    print(f"Lowest I_max/I_sig ratio index: {min_idx}, Ratio: {ratios[min_idx]:.4f}")
    
    best_I = intensities[min_idx]
    # Retrieve input fields again
    _, _, A_best, phi_best = run_simulation_batch(df[min_idx:min_idx+1], sp_size, mode='amp_info', seed=42)
    
    plot_qq(best_I, "Task1_QQPlot.png")
    generate_scientific_plots(best_I, A_best, phi_best, filename_prefix="Task1_Analysis")
    
    # --- Iteration (2, 4, 8, 16) ---
    print("\n--- ANALYSIS 1: Granularity Trend (I_max / I_1/3) ---")
    sp_sizes = [4, 8, 12, 16]
    min_ratios = []
    max_ratios = []
    
    for sp in sp_sizes:
        r, _, _, _ = run_simulation_batch(df, sp, mode='amp_info', seed=42)
        min_ratios.append(np.min(r))
        max_ratios.append(np.max(r))
        print(f"SP {sp}: Min Ratio={np.min(r):.3f}, Max Ratio={np.max(r):.3f}")
        
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(8)))
    plt.plot(sp_sizes, min_ratios, 'o-', color='#27AE60', label='Min ($I_{max}/I_{1/3}$)')
    plt.plot(sp_sizes, max_ratios, 's-', color='#C0392B', label='Max ($I_{max}/I_{1/3}$)')
    plt.xlabel('Superpixel Granularity')
    plt.ylabel(r'$I_{max} / I_{1/3}$')
    plt.title('Rogue Wave Ratio vs Granularity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Task1_GranularityTrend.png", dpi=300)
    plt.show()

    # --- ANALYSIS 2: Superpixel=8, Info=Phase, Random=Amp ---
    print("\n--- ANALYSIS 2: Superpixel=8, Info=Phase, Random=Amp ---")
    
    ratios_2, intensities_2, _, _ = run_simulation_batch(df, 8, mode='phase_info', seed=42)
    
    min_idx_2 = np.argmin(ratios_2)
    print(f"Lowest I_max/I_sig ratio index: {min_idx_2}, Ratio: {ratios_2[min_idx_2]:.4f}")
    
    best_I_2 = intensities_2[min_idx_2]
    _, _, A_best_2, phi_best_2 = run_simulation_batch(df[min_idx_2:min_idx_2+1], 8, mode='phase_info', seed=42)
    
    plot_qq(best_I_2, "Task2_QQPlot.png")
    generate_scientific_plots(best_I_2, A_best_2, phi_best_2, filename_prefix="Task2_Analysis")
    
    print("\nAll analyses completed.")

if __name__ == "__main__":
    main()