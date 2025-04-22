from utils.imports import *
from utils.functions import *
from datasets.datasets import *
from models.FeatAnalysisViT import *
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
    num_proc=4,
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
channels_in = rot_images[0].shape[0]

img_size = 160
channels_in = 3
patch_size = 16
batch_size = 32
embed_dim = 512
num_layers = 8
num_heads = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if logged:
  print("Already logged in wandb!")
else:
  print("logging in wandb!")
  logged = login_in_wandb(logged)
  print("logged in wandb!")

print("Downloading Pre-Trained model")
try:
    run = wandb.init()
    artifact = run.use_artifact('YOUR_WANDB_USERNAME/vit-self-supervised-rotation/ROT_ViT_1.pth:v0', type='model')
    model_dir = artifact.download()
    model_path = f"{model_dir}/ROT_ViT_1.pth"

    net = FeatAnalysisViT(image_size=img_size, channels_in=channels_in, patch_size=patch_size,
             hidden_size=embed_dim, num_classes=4, num_layers=num_layers, num_heads=num_heads)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model successfully downloaded!")

except Exception as e:
    print(f"An error occurred during the download of the model: {e}")


net = net.to(device)
num_classes_imagenette = 10
hidden_size = net.hidden_size
num_layers = len(net.blocks)
layer_accuracies = []

base_lr = 1.5e-4
min_lr = 1e-6
weight_decay = 0.05
num_epochs_ft = 20
patience = 3
delta = 0.001
early_stopping = True


for layer_idx in range(num_layers + 1):
    print(f"\n=== Analysis of Layer {layer_idx} ===")
    layer_name = "Embedding" if layer_idx == 0 else f"Transformer Block {layer_idx}"
    print(f"Extracting features from: {layer_name}")

    feature_model = FeatureClassifier(net, layer_idx, num_classes_imagenette).to(device)

    optimizer_lp = optim.AdamW(feature_model.classifier.parameters(),
                             lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    criterion_lp = nn.CrossEntropyLoss()
    scheduler_lp = CosineAnnealingLR(optimizer_lp, T_max=num_epochs_ft, eta_min=min_lr)

    run_name = f"Features_Layer_{layer_idx}_{layer_name}"
    layer_run = setup_wandb(project_name="vit-feature-analysis", run_name=run_name)

    accuracy_train_cls, loss_train_cls, accuracy_val_cls, loss_val_cls = train_wandb(
        feature_model, criterion_lp, optimizer_lp, scheduler_lp, num_epochs_ft,
        device, 'classification', trainloader, valloader,
        early_stopping=early_stopping, patience=patience, delta=delta, use_wandb=True
    )

    best_val_acc = max(accuracy_val_cls)
    layer_accuracies.append(best_val_acc)

    print(f"Layer {layer_idx} ({layer_name}) - Best accuracy validation: {best_val_acc:.2f}%")

    wandb.finish()
