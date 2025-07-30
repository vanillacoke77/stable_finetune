import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import (
    StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL, DiffusionPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Memory optimization settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set up paths
dataset_path = "./dataset/nih_dataset_capstone-main/dataset"  # Change this if needed
output_dir = "./sd_finetuned"
os.makedirs(output_dir, exist_ok=True)

# Verify dataset
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
print(f"Found {len(image_files)} PNG images in dataset")

# Custom Dataset Class
class NIHXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create dataset and dataloader
batch_size = 4  # Adjust based on your GPU memory
dataset = NIHXRayDataset(dataset_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load model components (keep weights in float32)
model_id = "stabilityai/stable-diffusion-2-base"

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Memory optimizations
unet.enable_gradient_checkpointing()
for param in text_encoder.parameters():
    param.requires_grad = False
for param in vae.parameters():
    param.requires_grad = False

# Training parameters
num_epochs = 50
learning_rate = 1e-5
gradient_accumulation_steps = 4

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

scaler = torch.amp.GradScaler(device='cuda')

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs,
)

progress_bar = tqdm(range(num_epochs * len(dataloader)))
global_step = 0

for epoch in range(num_epochs):
    unet.train()
    for step, batch in enumerate(dataloader):
        clean_images = batch.to(device).to(torch.float32)  # keep float32 for model input

        with torch.no_grad():
            latents = vae.encode(clean_images).latent_dist.sample().to(device)
            latents = latents * 0.18215  # Stable Diffusion latent scale

        noise = torch.randn_like(latents).to(device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        input_ids = tokenizer(
            [""] * latents.shape[0],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        with torch.amp.autocast(device_type='cuda'):
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step
        }
        progress_bar.set_postfix(**logs)
        global_step += 1

        torch.cuda.empty_cache()

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        unet.save_pretrained(os.path.join(save_path, "unet"))

        with torch.no_grad():
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                unet=unet,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                safety_checker=None,
            ).to(device)

            generator = torch.Generator(device=device).manual_seed(42)
            images = pipeline(
                ["chest x-ray"] * 2,
                generator=generator,
                num_inference_steps=30,
                height=128,
                width=128
            ).images

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            for i, img in enumerate(images):
                axs[i].imshow(img)
                axs[i].axis('off')
            plt.savefig(os.path.join(save_path, f"samples_epoch_{epoch+1}.png"))
            plt.close()

            del pipeline
            torch.cuda.empty_cache()

final_save_path = os.path.join(output_dir, "final_model")
os.makedirs(final_save_path, exist_ok=True)
unet.save_pretrained(os.path.join(final_save_path, "unet"))

print("Training complete!")
