from utils.imports import *
from utils.functions import *
from datasets.datasets import *
from models.ViT import *
logged_wandb = False
logged_hf = False

WANDB_API_KEY=''
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
original_images, rot_images, rot_labels, cls_labels = next(dataiter)

img_size = rot_images[0].shape[1]
channels_in = rot_images[2].shape[0]
embed_dim = 512
num_layers = 8
num_heads = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net_not_pre_trained = ViT(image_size=img_size, channels_in=channels_in, patch_size=patch_size, hidden_size=embed_dim, num_classes=10, num_layers=num_layers, num_heads=num_heads)
net_not_pre_trained = net_not_pre_trained.to(device)

run = setup_wandb(project_name="vit-self-supervised-rotation", run_name="ViT_not_pre-trained")

task_cls = 'classification'
early_stopping = True
patience = 5
delta = 0.001
base_lr = 1e-5 
min_lr = 1e-6 #min value for cosine annealing
weight_decay = 0.05
momentum = 0.9
num_epochs = 50
warmup_epochs = 10
num_epochs_ft = 20
optimizer = optim.AdamW(net_not_pre_trained.parameters(), lr=base_lr, weight_decay=weight_decay ,betas=(0.9, 0.95))
criterion = nn.CrossEntropyLoss()
scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda) #phase: warmup
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr) #phase: cosine deacy
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

# Sposta il modello sul device
net = net_not_pre_trained.to(device)

# Training del linear probing
accuracy_train_rot_ViT, loss_train_rot_ViT, accuracy_test_rot_ViT, loss_test_rot_ViT = train_wandb(
    net_not_pre_trained, criterion, optimizer, scheduler, num_epochs, device, task_cls, trainloader, valloader,
    early_stopping=early_stopping, patience=patience, delta=delta, use_wandb=True)

# Salva il modello linear probed su wandb
linear_probed_model_path = save_model_to_wandb(net, "not_pre_trained_ViT.pth", run)