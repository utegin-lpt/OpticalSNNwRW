import numpy as np
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
samples = np.load("BreastMNIST/images.npy")/255
labels = np.load("BreastMNIST/breast_labels.npy")
num_classes = len(np.unique(labels))

print(f"Original data shape: {samples.shape}")

# Parameters
xres, yres = 8e-6, 8e-6
sq_x = 480
spacewidth = sq_x * xres
spacelength = sq_x * yres
beam_width_x = 224 * xres  
beam_width_y = 224 * yres
x_fwhm, y_fwhm = beam_width_x / 2.355, beam_width_y / 2.355

x_coords = torch.linspace(-spacewidth*0.5, spacewidth*0.5, int(spacewidth/xres))
y_coords = torch.linspace(-spacelength*0.5, spacelength*0.5, int(spacelength/yres))
Y_grid, X_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

# Pre-calculate Gaussian field A
A_field = torch.sqrt((1 / (math.pi * x_fwhm * y_fwhm)) * torch.exp(-((X_grid**2) / (2 * beam_width_x**2) + 
                                 (Y_grid**2) / (2 * beam_width_y**2))))


class PhasePattern(nn.Module):
    def __init__(self, height, width, superpixel_size, pattern='checkerboard'):
        super().__init__()
        self.height = height
        self.width = width
        self.superpixel_size = superpixel_size
        self.pattern = pattern
        
        if pattern == 'checkerboard':
            grid_h = int(height / superpixel_size)
            grid_w = int(width / superpixel_size)
            self.phase_values = nn.Parameter(torch.rand(grid_h, grid_w) * 2 * math.pi)
            
        elif pattern == 'line':
            grid_h = int(height / superpixel_size)
            self.phase_values = nn.Parameter(torch.rand(grid_h, 1) * 2 * math.pi)
            
        elif pattern == 'vertical_line':
            grid_w = int(width / superpixel_size)
            self.phase_values = nn.Parameter(torch.rand(1, grid_w) * 2 * math.pi)
            
        elif pattern == 'circle':
            diameter = height
            radius = diameter / 2
            num_rings = int(radius / superpixel_size)
            self.phase_values = nn.Parameter(torch.rand(num_rings) * 2 * math.pi)
            self.diameter = diameter
            self.radius = radius
            self.num_rings = num_rings
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def forward(self):
        if self.pattern == 'checkerboard':
            phase = torch.repeat_interleave(self.phase_values, self.superpixel_size, dim=0)
            phase = torch.repeat_interleave(phase, self.superpixel_size, dim=1)
            phase = phase[:self.height, :self.width]
            
        elif self.pattern == 'line':
            phase = torch.repeat_interleave(self.phase_values, self.superpixel_size, dim=0)
            phase = phase.repeat(1, self.width)
            phase = phase[:self.height, :self.width]
            
        elif self.pattern == 'vertical_line':
            phase = torch.repeat_interleave(self.phase_values, self.superpixel_size, dim=1)
            phase = phase.repeat(self.height, 1)
            phase = phase[:self.height, :self.width]
            
        elif self.pattern == 'circle':
            y, x = torch.meshgrid(
                torch.arange(self.diameter, device=self.phase_values.device),
                torch.arange(self.diameter, device=self.phase_values.device),
                indexing='ij'
            )
            center = self.diameter / 2
            r = torch.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
            r_normalized = r / self.radius
            
            ring_indices = (r_normalized * self.num_rings).long()
            ring_indices = torch.clamp(ring_indices, 0, self.num_rings - 1)
            
            phase = self.phase_values[ring_indices]
            mask = (r <= self.radius).float()
            phase = phase * mask
            
            if self.width > self.diameter:
                pad_total = self.width - self.diameter
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                phase = F.pad(phase, (pad_left, pad_right, 0, 0), mode='constant', value=0)
        
        return phase


class RogueWaveThreshold(nn.Module):
    def __init__(self, steepness=10.0, sigma_multiplier=8.0):
        super().__init__()
        self.steepness = steepness
        self.sigma_multiplier = sigma_multiplier
    
    def compute_rogue_threshold(self, intensity):
        B, H, W = intensity.shape
        flat_intensity = intensity.view(B, -1)
        N = flat_intensity.shape[1]
        k = max(1, N // 3)
        
        top_values, _ = torch.topk(flat_intensity, k, dim=1)
        I_significant = top_values.mean(dim=1, keepdim=True)
        I_RW = 2 * I_significant
        
        return I_RW.view(B, 1, 1)
    
    
    def forward(self, intensity):
        threshold = self.compute_rogue_threshold(intensity)
        soft_mask = torch.sigmoid(self.steepness * (intensity - threshold))
        
        gated_intensity = soft_mask #Get the binary mask
        
        return gated_intensity, threshold, soft_mask


class OpticalNet(nn.Module):
    def __init__(self, num_classes, wavelength=635e-9, 
                 input_size=(224, 224), target_size=(400, 400),
                 superpixel_size=40, pool_size=16, phase_pattern='checkerboard',
                 threshold_method='rogue_wave', steepness=10.0):
        super().__init__()
        
        self.wavelength = wavelength
        self.resolution = [8e-6, 8e-6]
        self.num_classes = num_classes
        self.target_size = target_size
        
        self.pad_h = (target_size[0] - input_size[0]) // 2
        self.pad_w = (target_size[1] - input_size[1]) // 2
        
        self.register_buffer('source_field', A_field)
        
        self.phase_generator = PhasePattern(
            input_size[0], input_size[1], superpixel_size, pattern=phase_pattern
        )
        
        self.rogue_threshold = RogueWaveThreshold(
            method=threshold_method, 
            steepness=steepness
        )
        
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        
        feat_h = target_size[0] // pool_size
        feat_w = target_size[1] // pool_size
        self.feature_dim = feat_h * feat_w 
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)


    def pad_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  
        x_padded = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), 
                         mode='constant', value=0)
        return x_padded
    
    def forward(self, input_field, return_debug=False):
        B = input_field.shape[0]
        
        input_field_padded = self.pad_input(input_field)
        phi = self.phase_generator()
        phi_padded = self.pad_input(phi)
        
        modulated = self.source_field * input_field_padded * torch.exp(1j *phi_padded)
        
        # Use utils.propagation_ASM as requested
        U = utils.propagation_ASM(
            modulated,
            self.resolution, 
            self.wavelength, 
            40e-2
        )
        
        intensity = torch.abs(U) ** 2
        intensity = intensity.squeeze(1)        
        gated_intensity, threshold, soft_mask = self.rogue_threshold(intensity)
        
        x = gated_intensity.unsqueeze(1)
        pooled_features = self.pool(x)
        
        flat_features = pooled_features.flatten(1)
        
        logits = self.classifier(flat_features)
        
        if return_debug:
            return logits, {
                'input_padded': input_field_padded,
                'intensity': intensity,
                'gated_intensity': gated_intensity,
                'threshold': threshold,
                'soft_mask': soft_mask,
                'phase': phi,
                'pooled_features': pooled_features
            }
        
        return logits

x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

batch_size = 20
train_dataset = TensorDataset(x_train_t, y_train_t)
test_dataset = TensorDataset(x_test_t, y_test_t)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = OpticalNet(
    num_classes=num_classes,
    input_size=(224, 224), 
    target_size=(sq_x, sq_x),
    superpixel_size=8,
    pool_size=8,
    phase_pattern='checkerboard',
    threshold_method='rogue_wave',
    steepness=100.0
).to(device)

number_of_epoch = 400
loss_values_train = []
loss_values_test = []
acc_values_train = []
acc_values_test = []
lr_history = []

min_loss = float('inf')
best_epoch = 0
patience_counter = 0
early_stop_patience = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=number_of_epoch)
criterion = nn.CrossEntropyLoss()

print(f"\nModel Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Initial LR: {optimizer.param_groups[0]['lr']}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

for epoch in range(number_of_epoch):
    # --- Training ---
    model.train()
    epoch_loss_train = 0
    correct_train = 0
    total_train = 0
    
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        
        if y.ndim > 1:
            y = y.squeeze()
        
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss_train += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total_train += y.size(0)
        correct_train += (predicted == y).sum().item()
        
    avg_train_loss = epoch_loss_train / len(train_dataloader)
    train_acc = 100 * correct_train / total_train
    loss_values_train.append(avg_train_loss)
    acc_values_train.append(train_acc)
    
    # --- Validation ---
    model.eval()
    epoch_loss_test = 0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            
            if y.ndim > 1:
                y = y.squeeze()
                
            logits = model(x)
            loss = criterion(logits, y)
            
            epoch_loss_test += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total_test += y.size(0)
            correct_test += (predicted == y).sum().item()
    
    avg_test_loss = epoch_loss_test / len(test_dataloader)
    test_acc = 100 * correct_test / total_test
    loss_values_test.append(avg_test_loss)
    acc_values_test.append(test_acc)
    
    # Update learning rate
    scheduler.step()
    lr_history.append(optimizer.param_groups[0]['lr'])
    
    print(f'\rEpoch {epoch+1}/{number_of_epoch} | '
          f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | '
          f'Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.2f}% | '
          f'LR: {optimizer.param_groups[0]["lr"]:.2e}', end='')
    
    # Save best model
    if avg_test_loss < min_loss:
        min_loss = avg_test_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }, 'best_model')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f'\n\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)')
            break

print("\n\nTraining complete!")
print(f"Best model saved at epoch {best_epoch} with test loss: {min_loss:.4f}")

# Load best model for final evaluation
checkpoint = torch.load('best_model')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss
axes[0, 0].plot(loss_values_train, label='Train Loss', alpha=0.7)
axes[0, 0].plot(loss_values_test, label='Test Loss', alpha=0.7)
axes[0, 0].axvline(best_epoch-1, color='red', linestyle='--', alpha=0.5, label='Best Model')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].set_title('Loss Curve')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[0, 1].plot(acc_values_train, label='Train Acc', alpha=0.7)
axes[0, 1].plot(acc_values_test, label='Test Acc', alpha=0.7)
axes[0, 1].axvline(best_epoch-1, color='red', linestyle='--', alpha=0.5, label='Best Model')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].set_title('Accuracy Curve')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
axes[1, 0].plot(lr_history, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Overfitting Gap
gap = np.array(acc_values_train) - np.array(acc_values_test)
axes[1, 1].plot(gap, color='orange', alpha=0.7)
axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Train Acc - Test Acc (%)')
axes[1, 1].set_title('Overfitting Gap')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results_.png', dpi=300)

# ============================================================================
# FINAL VISUALIZATION
# ============================================================================

# Load best model checkpoint for visualization
checkpoint = torch.load('best_model_')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nUsing best model from epoch {checkpoint['epoch']+1} for visualization")

model.eval()
with torch.no_grad():
    x_sample, y_sample = next(iter(test_dataloader))
    x_sample = x_sample.to(device)
    y_sample = y_sample.to(device)
    
    logits, debug = model(x_sample, return_debug=True)
    preds = torch.argmax(logits, dim=1)
    
    idx = 0
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1
    img = axes[0, 0].imshow(x_sample[idx].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Original Input (224x224)")
    axes[0, 0].axis('off')
    
    img = axes[0, 1].imshow(debug['input_padded'][idx, 0].cpu().numpy(), cmap='gray')
    axes[0, 1].set_title("Padded Input (480×480)")
    fig.colorbar(img, ax=axes[0, 1])
    axes[0, 1].axis('off')
    
    img = axes[0, 2].imshow(model.pad_input(debug['phase']).cpu().numpy(), 
                            cmap='twilight', vmin=0, vmax=2*np.pi)
    axes[0, 2].set_title("Learned Phase Pattern")
    fig.colorbar(img, ax=axes[0, 2])
    axes[0, 2].axis('off')
    
    img = axes[0, 3].imshow(debug['intensity'][idx].cpu().numpy(), cmap='hot')
    axes[0, 3].set_title("Raw Output Intensity")
    fig.colorbar(img, ax=axes[0, 3])
    axes[0, 3].axis('off')
    
    # Row 2
    img = axes[1, 0].imshow(debug['soft_mask'][idx].cpu().numpy(), cmap='hot')
    axes[1, 0].set_title(f"Rogue Wave Mask\n(Th={debug['threshold'][idx].item():.2e})")
    fig.colorbar(img, ax=axes[1, 0])
    axes[1, 0].axis('off')
    
    img = axes[1, 1].imshow(debug['gated_intensity'][idx].cpu().numpy(), cmap='hot')
    axes[1, 1].set_title("Gated Intensity (Rogue Waves)")
    fig.colorbar(img, ax=axes[1, 1])
    axes[1, 1].axis('off')
    
    # Show Pooled Features (New Visualization)
    pooled_img = debug['pooled_features'][idx, 0].cpu().numpy()
    img = axes[1, 2].imshow(pooled_img, cmap='hot')
    axes[1, 2].set_title(f"Pooled Features (25x25)")
    fig.colorbar(img, ax=axes[1, 2])
    axes[1, 2].axis('off')
    
    # Class energies / Logits
    class_energies = logits[idx].detach().cpu().numpy()
    axes[1, 3].bar(range(num_classes), class_energies, 
                   color=['blue', 'red'][:num_classes])
    axes[1, 3].set_title(
        f"Output Logits\nPred: {preds[idx].item()} | GT: {y_sample[idx].item()}"
    )
    axes[1, 3].set_xlabel('Class')
    axes[1, 3].set_ylabel('Logit Value')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization_.png', dpi=300)

print(f"\nSample prediction accuracy: {(preds == y_sample).float().mean().item() * 100:.2f}%")
print(f"Best test accuracy: {checkpoint['test_acc']:.2f}%")