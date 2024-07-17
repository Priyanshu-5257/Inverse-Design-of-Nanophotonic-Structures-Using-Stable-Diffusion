import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
T = 300

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
def load_data(csv_path):
    return pd.read_csv(csv_path)

def prepare_data(df):
    train_df = df.sample(frac=0.95, random_state=42)
    test_df = df.drop(train_df.index)
    return train_df, test_df

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.dataframe.iloc[idx, 1:801].values.tolist(), dtype=torch.float32)
        image_name = self.dataframe.iloc[idx, :1].values[0].replace("-Excel.mat", "-colorprops.png").replace("nm_", "nm")
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return features, image

# Model definitions
class SpectraEncoder(nn.Module):
    def __init__(self):
        super(SpectraEncoder, self).__init__()
        self.spectral_embedding = nn.Sequential(
            nn.Linear(in_features=800, out_features=784),
            nn.ReLU(),
            nn.Linear(in_features=784, out_features=784),
            nn.Tanh(),
        )

    def forward(self, spectra):
        return self.spectral_embedding(spectra)

def create_unet_model():
    return UNet2DConditionModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(128, 256, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        encoder_hid_dim=784
    )

# Diffusion process functions
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Loss function
def get_loss(model, x_0, spectra, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, spectra).sample
    return F.l1_loss(noise, noise_pred)

# Sampling and visualization functions
@torch.no_grad()
def sample_timestep(x, spectra, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, spectra).sample / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_plot_image(real_img, spectra):
    real_img = real_img.to(device)
    spectra = spectra.to(device)
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, spectra, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().to("cpu").squeeze(0))
    plt.show()            
    show_tensor_image(real_img)
    plt.show()

# Training function
def train(model, spectra_encoder, train_dataloader, test_df, epochs):
    optimizer = Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = batch[1].to(device)
            spectra = spectra_encoder(batch[0].unsqueeze(1).to(device))
            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            loss = get_loss(model, images, spectra, t)
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0 and step % 50 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                if epoch % 5 == 0:
                    sample = test_df.sample(1)
                    spectra = sample.iloc[:, 1:801].values
                    spectra = spectra_encoder(torch.tensor(spectra, dtype=torch.float32).unsqueeze(1).to(device))
                    
                    real_img_name = sample.iloc[:, 0].values
                    real_img_path = os.path.join('/kaggle/working/', real_img_name[0].replace("-Excel.mat","-colorprops.png").replace("nm_","nm"))
                    real_img = Image.open(real_img_path).convert('RGB')
                    sample_plot_image(transform(real_img).unsqueeze(0), spectra)

# Main execution
if __name__ == "__main__":
    # Data preparation
    df = load_data("/kaggle/working/Multiclass_Metasurface_InverseDesign/Training_Data/absorptionData_HybridGAN.csv")
    train_df, test_df = prepare_data(df)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    custom_dataset = CustomDataset(dataframe=train_df, image_dir='/kaggle/working/', transform=transform)
    train_dataloader = DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model initialization
    spectra_encoder = SpectraEncoder().to(device)
    model = create_unet_model().to(device)

    # Diffusion process setup
    betas = linear_beta_schedule(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Training
    train(model, spectra_encoder, train_dataloader, test_df, epochs=500)
