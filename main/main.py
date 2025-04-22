from utils.imports import *
from utils.functions import *
from datasets.datasets import *
from models.ViT import *
logged_wandb = False
logged_hf = False


WANDB_API_KEY=''
HF_API_KEY=''
WANDB_USERNAME=''

#PRE-TRAIN
logged_hf = login_to_huggingface(logged_hf, HF_API_KEY)
dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
print("done")
validation_set = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
print("done")
logged = login_in_wandb(logged_wandb, WANDB_API_KEY)


batch_size = 32
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

train_data = dataset
val_data = validation_set
trainset = StreamingRotation2(train_data, transform=transform_train)
valset = StreamingRotation(val_data, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         collate_fn=rotation_collate, num_workers=2)
valloader = DataLoader(valset, batch_size=batch_size, pin_memory=True)

dataiter = iter(trainloader)
original_images, rot_images, rot_labels, cls_labels = next(dataiter)

img_size = rot_images[0].shape[1]
channels_in = rot_images[0].shape[0]
classes_ft = 4
task_rot='rotation'
embed_dim = 512
num_layers = 8
num_heads = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run = setup_wandb(project_name="vit-self-supervised-rotation", run_name="Pre_train_ROT_ViT")

#VIt inizialization
net = ViT(image_size=img_size, channels_in=channels_in, patch_size=patch_size, hidden_size=embed_dim, num_classes=classes_ft, num_layers=num_layers, num_heads=num_heads)
net = net.to(device)

#PRE-Train Params
early_stopping = True
patience = 5
delta = 0.001
base_lr = 1.5e-4
min_lr = 1e-6 
weight_decay = 0.05
num_epochs = 5
steps_per_epoch = 40000
total_steps = steps_per_epoch * num_epochs
warmup_steps = 20000  #more or less half of an epoch
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
criterion = nn.CrossEntropyLoss()
scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

# Training
accuracy_train_rot_ViT, loss_train_rot_ViT, accuracy_test_rot_ViT, loss_test_rot_ViT = pre_train_wandb(
    net, criterion, optimizer, scheduler, num_epochs, device, task_rot, trainloader, valloader, run,
    early_stopping=early_stopping, patience=patience, delta=delta, use_wandb=True)

#LINEAR PROBING and FINE TUNING
ds = load_dataset("frgfm/imagenette", "320px")

ds = ds.map(check_image_channels)
problematic_ds = ds.filter(lambda x: x['is_problematic'])["train"]
print(f"Found {len(problematic_ds)} not RGB images")
ds_filtered = ds.map(
    convert_to_rgb,
    num_proc=4,
    desc="Convert non-RGB images to RGB"
)

batch_size = 32

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


#LINEAR PROBING
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
    model_path = f"{model_dir}/ROT_ViT_1.pth"

    net = ViT(image_size=img_size, channels_in=channels_in, patch_size=patch_size,
             hidden_size=embed_dim, num_classes=4, num_layers=num_layers, num_heads=num_heads)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model successfully downloaded")

except Exception as e:
    print(f"An errorr occurred during the download of the model: {e}")


net = net.to(device)

#then we subsitute the calssification head:
num_classes_imagenette = 10  # imagenette has 10 classes
hidden_size = net.hidden_size
net.fc_out = nn.Linear(hidden_size, num_classes_imagenette).to(device)
#freeze
param_to_train = 0
param_total = 0
for name, param in net.named_parameters():
    param_total += param.numel()
    if 'fc_out' not in name:
        param.requires_grad = False
    else:
        param_to_train += param.numel()
        print(f"Active Parameters: {name}")

print(f"Training of {param_to_train} parameters over {param_total} total ({100*param_to_train/param_total:.2f}%)")

run = setup_wandb(project_name="vit-self-supervised-rotation", run_name="LP_of_ROT_VIT_1")

base_lr = 1.5e-4
min_lr = 1e-6
weight_decay = 0.05
num_epochs_ft = 20
patience = 3
delta = 0.001
early_stopping = True

optimizer_lp = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                         lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
criterion_lp = nn.CrossEntropyLoss()
scheduler_lp = CosineAnnealingLR(optimizer_lp, T_max=num_epochs_ft, eta_min=min_lr)

print("Starting Linear Probing...")
accuracy_train_cls, loss_train_cls, accuracy_val_cls, loss_val_cls = train_wandb(
    net, criterion_lp, optimizer_lp, scheduler_lp, num_epochs_ft,
    device, 'classification', trainloader, valloader, early_stopping=early_stopping, patience= patience, delta=delta, use_wandb=True
)
linear_probed_model_path = save_model_to_wandb(net, "lp_ROT_ViT_1.pth", run)

#FINE TUNING 1: CH + L7
if logged:
  print("Already logged in wandb!")
else:
  print("logging in wandb!")
  logged = login_in_wandb(logged)
  print("logged in wandb!")

print("Downloading LP-Model from wandb")
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

#Fine Tuning 2: CH + L7 + L6
if logged:
  print("Already logged in wandb!")
else:
  print("logging in wandb!")
  logged = login_in_wandb(logged)
  print("logged in wandb!")

print("Downloading Fine-yuned model from wandb")
try:
    run = wandb.init()
    artifact = run.use_artifact(f'{WANDB_USERNAME}/vit-self-supervised-rotation/ROT_ViT_1.pth:v0', type='model')
    model_dir = artifact.download()
    model_path = f"{model_dir}/FT-fc7_ROT_ViT_1.pth"

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
    if 'fc_out' in name or 'blocks.7' in name or 'blocks.6' in name:
        param.requires_grad = True
        param_to_train += param.numel()
        print(f"Active parameters: {name}")
    else:
        param.requires_grad = False

print(f"Training of {param_to_train} parameters over {param_total} total ({100*param_to_train/param_total:.2f}%)")

run = setup_wandb(project_name="vit-self-supervised-rotation", run_name="FT_fc76_of_ROT_VIT_1")
num_epochs_ft = 10
patience = 2
delta = 0.005


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

linear_probed_model_path = save_model_to_wandb(net, "FT-fc76_ROT_ViT_1.pth", run)