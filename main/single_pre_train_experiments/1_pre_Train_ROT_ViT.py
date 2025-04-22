from utils.imports import *
from utils.functions import *
from datasets.datasets import *
from models.ViT import *
logged_wandb = False
logged_hf = False

#use your API key here
WANDB_API_KEY=''
HF_API_KEY=''
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

#Train Params
early_stopping = True
patience = 5
delta = 0.001
base_lr = 1.5e-4
min_lr = 1e-6 
weight_decay = 0.05
num_epochs = 5  #reduced for ImageNet
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