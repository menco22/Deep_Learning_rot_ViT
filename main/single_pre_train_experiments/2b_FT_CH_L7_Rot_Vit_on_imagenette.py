from utils.imports import *
from utils.functions import *
from datasets.datasets import *
from models.ViT import *
logged_wandb = False
logged_hf = False

WANDB_API_KEY=''
WANDB_USERNAME=''
logged = login_in_wandb(logged_wandb, WANDB_API_KEY)

ds = load_dataset("frgfm/imagenette", "320px")
logged = login_in_wandb(logged)

#Note: Imagenette has some images that are not RGB => we need to convert them
ds = ds.map(check_image_channels)
problematic_ds = ds.filter(lambda x: x['is_problematic'])["train"]
print(f"Found {len(problematic_ds)} not RGB images")
ds_filtered = ds.map(
    convert_to_rgb,
    num_proc=4, #For parallel computation
    desc="Convert non-RGB images to RGB"
)

batch_size = 16
patch_size = 16

transform_train = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((160, 160), Image.LANCZOS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((160, 160), Image.LANCZOS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ds_filtered['train']
val_data = ds_filtered['validation']
trainset = customRotation2(train_data, transform=transform_train)
valset = customRotation(val_data, transform=transform_test)

# DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, collate_fn=rotation_collate, num_workers=2)

valloader = DataLoader(valset, batch_size=batch_size,
                       shuffle=False)
dataiter = iter(trainloader)
original_images, rot_images, rot_labels, cls_labels = next(dataiter) #rot_images is a batch of 32*4 images!

img_size = rot_images[0].shape[1]
channels_in = rot_images[2].shape[0]
embed_dim = 512
num_layers = 8
num_heads = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#now we need to download the previously pre-trained model
if logged:
  print("Already logged in wandb!")
else:
  print("logging in wandb!")
  logged = login_in_wandb(logged)
  print("logged in wandb!")

print("Downloading Pre-trained Model from wandb")
try:
    run = wandb.init()
    artifact = run.use_artifact(f'{WANDB_USERNAME}/vit-self-supervised-rotation/ROT_ViT_1.pth:v0', type='model')
    model_dir = artifact.download()
    model_path = f"{model_dir}/lp_ROT_ViT_1.pth"

    net = ViT(image_size=img_size, channels_in=channels_in, patch_size=patch_size,
             hidden_size=embed_dim, num_classes=10, num_layers=num_layers, num_heads=num_heads)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model successfully downloaded")

except Exception as e:
    print(f"An errorr occurred during the download of the model: {e}")

net = net.to(device)

#freeze
param_to_train = 0
param_total = 0
for name, param in net.named_parameters():
    param_total += param.numel()
    if 'fc_out' in name or 'blocks.7' in name:
        param.requires_grad = True
        param_to_train += param.numel()
        print(f"Active parameters: {name}")
    else:
        param.requires_grad = False

print(f"Training of {param_to_train} parameters over {param_total} total ({100*param_to_train/param_total:.2f}%)")

run = setup_wandb(project_name="vit-self-supervised-rotation", run_name="FT_fc7_of_ROT_VIT_1")
base_lr = 1.5e-4
min_lr = 1e-6
weight_decay = 0.05
num_epochs_ft = 20
patience = 3
delta = 0.001
early_stopping = True

optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                         lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
criterion_ft = nn.CrossEntropyLoss()
scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=num_epochs_ft, eta_min=min_lr)

net = net.to(device)

print("Starting Fine Tuning...")

accuracy_train_cls, loss_train_cls, accuracy_val_cls, loss_val_cls = train_wandb(
    net, criterion_ft, optimizer_ft, scheduler_ft, num_epochs_ft,
    device, 'classification', trainloader, valloader, early_stopping=early_stopping, patience= patience, delta=delta, use_wandb=True
)

linear_probed_model_path = save_model_to_wandb(net, "FT-fc7_ROT_ViT_1.pth", run)