import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os
import random
from matplotlib.backends.backend_pdf import PdfPages
import time
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm đếm số lượng tham số chi tiết
def count_params(model, header="Detailed Model Parameters"):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"\n{header}:")
    print(table)
    print(f"Total Trainable Parameters: {total_params}")
    return total_params

# Hàm đếm tham số không được sử dụng
def count_unused_params(model):
    unused_params = []
    unused_param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused_params.append((name, param.numel()))
            unused_param_count += param.numel()
    if unused_param_count > 0:
        print(f"\nUnused Parameters Detected: {unused_param_count}")
        unused_table = PrettyTable(["Unused Modules", "Parameters"])
        for name, size in unused_params:
            unused_table.add_row([name, size])
        print(unused_table)
    else:
        print("\nNo Unused Parameters Detected.")
    return unused_param_count

# Hàm Spline Linear tối ưu
class SplineLinear(nn.Module):
    def __init__(self, in_features, out_features, num_grids=12):
        super(SplineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.coeff = nn.Parameter(torch.randn(out_features, num_grids))
        self.grid = torch.linspace(-1, 1, num_grids).to(device)
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.proj(x)
        x = x.view(batch_size, self.out_features, 1)
        grid = self.grid.view(1, 1, self.num_grids).expand(batch_size, self.out_features, self.num_grids)
        diff = x - grid
        basis = torch.exp(-diff.pow(2))
        output = torch.einsum('bik,ik->bi', basis, self.coeff)
        return output

# Hàm cơ sở RBF
class RadialBasisFunction(nn.Module):
    def __init__(self, num_grids=12):
        super(RadialBasisFunction, self).__init__()
        self.num_grids = num_grids
        self.centers = nn.Parameter(torch.linspace(-1, 1, num_grids), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        centers = self.centers.view(1, 1, self.num_grids).expand(batch_size, x.size(1), self.num_grids)
        diff = x - centers
        return torch.exp(-torch.square(diff) / (2 * self.sigma ** 2))

# Mô hình FastKAN tối ưu cho Super-Resolution
class FastKAN_SR(nn.Module):
    def __init__(self, input_dim=64, output_dim=4096, num_grids=12, hidden_dim=256):
        super(FastKAN_SR, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids

        self.rbf_layer = RadialBasisFunction(num_grids=num_grids)
        
        self.spline_inner = SplineLinear(input_dim * num_grids, hidden_dim, num_grids=num_grids)
        self.bn_inner = nn.BatchNorm1d(hidden_dim)
        self.spline_outer = SplineLinear(hidden_dim, output_dim, num_grids=num_grids)
        
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim)

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        residual = self.residual(x)

        rbf_out = self.rbf_layer(x)
        rbf_out = rbf_out.view(batch_size, -1)

        hidden = self.spline_inner(rbf_out)
        hidden = self.bn_inner(hidden)
        hidden = F.relu(hidden)

        output = self.spline_outer(hidden)
        output = output + residual
        output = torch.sigmoid(output)
        return output.view(batch_size, 1, 64, 64)

# Dataset
class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, selected_digits):
        self.dataset = [data for data in dataset if data[1] in selected_digits]
        self.transform_low = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
        self.transform_high = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform_low(img), self.transform_high(img), label

def save_dataset(dataset, filename): torch.save(dataset, filename)
def load_dataset(filename): return torch.load(filename) if os.path.exists(filename) else None

def calculate_psnr_ssim(pred, target):
    pred = pred.detach().squeeze().cpu().numpy()
    target = target.detach().squeeze().cpu().numpy()
    return psnr(target, pred, data_range=1.0), ssim(target, pred, data_range=1.0)

def validate_fastkan_sr(model, val_loader):
    model.eval()
    total_psnr, total_ssim = 0, 0
    with torch.no_grad():
        for low_res, high_res, _ in val_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            with autocast('cuda'):
                output = model(low_res)
            psnr_value, ssim_value = calculate_psnr_ssim(output, high_res)
            total_psnr += psnr_value
            total_ssim += ssim_value
    return total_psnr / len(val_loader), total_ssim / len(val_loader)

def test_metrics_fastkan_sr(model, test_loader, output_metrics_path="./Output_MNIST/output_fastkan/test_metrics.txt"):
    model.eval()
    total_psnr, total_ssim = 0, 0
    with torch.no_grad():
        for low_res, high_res, _ in test_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            with autocast('cuda'):
                output = model(low_res)
            psnr_value, ssim_value = calculate_psnr_ssim(output, high_res)
            total_psnr += psnr_value
            total_ssim += ssim_value
    test_psnr = total_psnr / len(test_loader)
    test_ssim = total_ssim / len(test_loader)

    # Lưu Test PSNR và SSIM vào file
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, 'w') as f:
        f.write(f"Test PSNR: {test_psnr:.2f}\n")
        f.write(f"Test SSIM: {test_ssim:.4f}\n")
    print(f"Test metrics saved to {output_metrics_path}")

    return test_psnr, test_ssim

def train_fastkan_sr(model, train_loader, val_loader, epochs=25, lr=0.005, patience=5):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    total_train_time = 0
    epoch_losses, all_train_psnr, all_train_ssim = [], [], []
    all_val_psnr, all_val_ssim = [], []
    best_val_psnr = -float('inf')
    patience_counter = 0
    best_model_path = "best_model.pt"

    for epoch in range(epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0
        epoch_start_time = time.time()

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for low_res, high_res, _ in train_loader_tqdm:
            low_res, high_res = low_res.to(device), high_res.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                output = model(low_res)
                loss = criterion(output, high_res)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            psnr_value, ssim_value = calculate_psnr_ssim(output, high_res)
            total_psnr += psnr_value
            total_ssim += ssim_value
            train_loader_tqdm.set_postfix(loss=total_loss / (train_loader_tqdm.n + 1),
                                          psnr=total_psnr / (train_loader_tqdm.n + 1),
                                          ssim=total_ssim / (train_loader_tqdm.n + 1))

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        epoch_losses.append(avg_loss)
        all_train_psnr.append(avg_psnr)
        all_train_ssim.append(avg_ssim)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} | Train SSIM: {avg_ssim:.4f}")
        print(f"Epoch {epoch+1} Time: {epoch_time:.2f} seconds")

        val_psnr, val_ssim = validate_fastkan_sr(model, val_loader)
        all_val_psnr.append(val_psnr)
        all_val_ssim.append(val_ssim)
        print(f"Validation PSNR: {val_psnr:.2f} | Validation SSIM: {val_ssim:.4f}\n")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model with Validation PSNR: {best_val_psnr:.2f}")

    avg_train_psnr = np.mean(all_train_psnr)
    avg_train_ssim = np.mean(all_train_ssim)
    avg_val_psnr = np.mean(all_val_psnr)
    avg_val_ssim = np.mean(all_val_ssim)

    print(f"Total Training Time: {total_train_time:.2f} seconds")
    print(f"Average Train PSNR: {avg_train_psnr:.2f}")
    print(f"Average Train SSIM: {avg_train_ssim:.4f}")
    print(f"Average Validation PSNR: {avg_val_psnr:.2f}")
    print(f"Average Validation SSIM: {avg_val_ssim:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_train_psnr) + 1), all_train_psnr, label="Train PSNR")
    plt.plot(range(1, len(all_val_psnr) + 1), all_val_psnr, label="Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("Train vs Validation PSNR")
    plt.legend()
    plt.grid()
    plt.savefig("./Output_MNIST/output_fastkan/psnr_fastkan_plot.pdf")
    plt.close()

    return model, total_train_time

def test_fastkan_sr(model, test_loader, output_path="./Output_MNIST/output_fastkan/results_fastkan.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    seen_labels = set()
    test_start_time = time.time()
    fixed_samples = {}

    target_labels = [9, 7, 6, 1]
    with torch.no_grad():
        for low_res, high_res, label in test_loader:
            label = label.item()
            if label in target_labels and label not in seen_labels:
                seen_labels.add(label)
                fixed_samples[label] = (low_res, high_res)
                if len(seen_labels) == len(target_labels):
                    break

    with PdfPages(output_path) as pdf:
        with torch.no_grad():
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for col_idx, label in enumerate(target_labels):
                low_res, high_res = fixed_samples[label]
                low_res = low_res.to(device)
                with autocast('cuda'):
                    output = model(low_res)
                low_res, high_res, high_res_pred = low_res.cpu(), high_res.cpu(), output.cpu()

                axes[0, col_idx].imshow(low_res[0].squeeze(), cmap='gray')
                axes[0, col_idx].set_title(f"Low Res - {label}")
                axes[0, col_idx].axis('off')

                axes[1, col_idx].imshow(high_res_pred[0].squeeze(), cmap='gray')
                axes[1, col_idx].set_title(f"Super Res - {label}")
                axes[1, col_idx].axis('off')

                axes[2, col_idx].imshow(high_res[0].squeeze(), cmap='gray')
                axes[2, col_idx].set_title(f"High Res - {label}")
                axes[2, col_idx].axis('off')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    test_time = time.time() - test_start_time
    return test_time

def main():
    train_dataset_file = "./train_data_set/FastKAN/MNIST/train_dataset.pt"
    val_dataset_file = "./val_data_set/FastKAN/MNIST/val_dataset.pt"
    test_dataset_file = "./test_data_set/FastKAN/MNIST/test_dataset.pt"
    output_metrics_path = "./Output_MNIST/output_fastkan/test_metrics.txt"
    selected_digits = [9, 7, 6, 1]
    print("Selected Digits:", selected_digits)
    print(f"Digits selected: {', '.join(map(str, selected_digits))}")

    # Load MNIST
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True)
    full_train_dataset = SuperResolutionDataset(mnist_dataset, selected_digits)
    
    # Kiểm tra số lượng mẫu trước khi chia
    total_samples = len(full_train_dataset)
    print(f"Total samples for digits {selected_digits}: {total_samples}")
    if total_samples < 6000:
        print(f"Warning: Not enough samples ({total_samples} < 6000). Adjusting split.")
        train_size = int(0.7 * total_samples)  # 70% train
        val_size = int(0.15 * total_samples)  # 15% validation
        test_size = total_samples - train_size - val_size  # 15% test
    else:
        train_size, val_size, test_size = 4000, 1000, 1000

    indices = random.sample(range(total_samples), train_size + val_size + test_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    save_dataset(train_dataset, train_dataset_file)
    save_dataset(val_dataset, val_dataset_file)

    mnist_test_dataset = datasets.MNIST(root="./data", train=False, download=True)
    full_test_dataset = SuperResolutionDataset(mnist_test_dataset, selected_digits)
    test_samples = len(full_test_dataset)
    print(f"Total test samples: {test_samples}")
    test_dataset = Subset(full_test_dataset, random.sample(range(test_samples), min(1000, test_samples)))
    save_dataset(test_dataset, test_dataset_file)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = FastKAN_SR()
    total_params = count_params(model, header="Model Parameters Before Training")

    trained_model, train_time = train_fastkan_sr(model, train_loader, val_loader, epochs=25, patience=5)

    print("\nChecking Parameters After Training:")
    count_params(trained_model, header="Model Parameters After Training")
    count_unused_params(trained_model)

    test_time = test_fastkan_sr(trained_model, test_loader)
    test_psnr, test_ssim = test_metrics_fastkan_sr(trained_model, test_loader, output_metrics_path)
    print(f"\nTest PSNR: {test_psnr:.2f} | Test SSIM: {test_ssim:.4f}")

    total_time = train_time + test_time
    print(f"\nTotal Time (Training + Testing): {total_time:.2f} seconds")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")

if __name__ == "__main__":
    main()