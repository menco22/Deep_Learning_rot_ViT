from imports import *

#rotation function
def rotate_img(img, rot):
    if rot == 0:
        return img

    elif rot == 1:
        return torch.rot90(img, 1, [1, 2])

    elif rot == 2:
        return torch.rot90(img, 2, [1, 2])

    elif rot == 3:
        return torch.rot90(img, 3, [1, 2])

    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')
    
#show images
def imshow(img):
    #unnormalize
    img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])(img)

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#validation
def run_test(net, testloader, criterion, device, task, validation=False):
    correct = 0
    total = 0
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in testloader:
            if len(batch) == 4:
                images, images_rotated, labels, cls_labels = batch
                
                if task == 'rotation':
                    images, labels = images_rotated.to(device), labels.to(device)
                elif task == 'classification':
                    images, labels = images.to(device), cls_labels.to(device)
                
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                batch_count += 1

    if batch_count > 0:
        avg_test_loss = total_loss / batch_count
        accuracy_test = 100 * correct / total if total > 0 else 0
    else:
        avg_test_loss = 0
        accuracy_test = 0
    
    if not validation:
        print('TESTING:')
        print(f'Accuracy of the model on the test images: {accuracy_test:.2f} %')
        print(f'Average loss on the test images: {avg_test_loss:.3f}')
    else:
        print('VALIDATION:')
        print(f'Accuracy of the model on the validation set: {accuracy_test:.2f} %')
        print(f'Average loss on the validation set: {avg_test_loss:.3f}')

    return accuracy_test, avg_test_loss

#OLD train function, not used in the code, it is an alternative to wandb
def train(net, criterion, optimizer, scheduler, num_epochs, device, task, trainloader, valloader):

    accuracy_train = []
    loss_train = []
    accuracy_val = []
    loss_val = []
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0
        start_time = time.time()

        net.train()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):

            if task == 'rotation':
                inputs, labels = imgs_rotated.to(device), rotation_label.to(device)
            elif task == 'classification':
                inputs, labels = imgs.to(device), cls_label.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(outputs, dim=1)
            print_freq = 100
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % print_freq == (print_freq - 1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()
        accuracy_train.append(100*running_correct / running_total)
        loss_train.append(running_loss / print_freq)
        net.eval()
        accuracy_temp, loss_temp = run_test(net, valloader, criterion, device, task, validation=True)
        accuracy_val.append(accuracy_temp)
        loss_val.append(loss_temp)
        scheduler.step()
    print('Finished Training')
    return accuracy_train, loss_train, accuracy_val, loss_val

#patch extractor for ViT, patch_size = 8
def extract_patches(image_tensor, patch_size):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded

#IMPORTANT!! Whithout collate, using only CIFAR10Rotation2, the customized __getItem__ will return list of images and labels, not tensors with the same dimensions
# => Dataloader can't concaneate all the concat of rot-images and rot-labels
def rotation_collate(batch):
    all_images = []
    all_rot_images = []
    all_rot_labels = []
    all_cls_labels = []

    for images, rot_images, rot_labels, cls_labels in batch:
        all_images.extend(images)
        all_rot_images.extend(rot_images)
        all_rot_labels.extend(rot_labels)
        all_cls_labels.extend(cls_labels)
    return torch.stack(all_images), torch.stack(all_rot_images), torch.tensor(all_rot_labels), torch.tensor(all_cls_labels)

#used for the scheduler of Vit
def warmup_lr_lambda(epoch, warmup_epochs):
    return min(1.0, epoch / warmup_epochs)

def setup_wandb(project_name="vit-self-supervised", run_name=None):
    run = wandb.init(project=project_name,
                     name=run_name,
                     config={
                         "architecture": "Vision Transformer",
                         "task": "Self-Supervised Learning (Rotation)",
                         "dataset": "TinyImageNet"
                     })

    return run

def login_in_wandb(logged, API_KEY):
  wandb.login(key=API_KEY)
  logged = True
  return logged

def login_to_huggingface(logged, API_KEY):
    login(token=API_KEY)
    logged = True
    return logged

def save_model_to_wandb(model, model_name, run=None):
    local_path = f"{model_name}"
    torch.save(model.state_dict(), local_path)

    if run is None:
        if wandb.run is None:
            print("NO Wandb session active. The model will be saved locally.")
            return local_path
        run = wandb.run

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=f"Model Vit Trained on: {model_name.split('_')[0]}"
    )
    artifact.add_file(local_path)
    run.log_artifact(artifact)

    print(f"Model Saved in Wandb as: {model_name}")
    return local_path

#use this function to pre_train the model, set use_wandb=False if you don't want to use wandb
def pre_train_wandb(net, criterion, optimizer, scheduler, num_epochs, device,
                task, trainloader, valloader, run, early_stopping,
                patience, delta, use_wandb=True):
    accuracy_train = []
    loss_train = []
    accuracy_val = []
    loss_val = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    global_step = 0

    for epoch in range(num_epochs):
        net.train()
        running_loss, running_correct, running_total = 0.0, 0.0, 0.0
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        start_time = time.time()
        batch_count = 0

        for batch in trainloader:
            if len(batch) == 4:
                imgs, imgs_rotated, rotation_label, cls_label = batch
                
                if task == 'rotation':
                    inputs, labels = imgs_rotated.to(device), rotation_label.to(device)
                elif task == 'classification':
                    inputs, labels = imgs.to(device), cls_label.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                scheduler.step()

                predicted = torch.argmax(outputs, dim=1)

                batch_size = labels.size(0)
                running_loss += loss.item()
                running_total += batch_size
                running_correct += (predicted == labels).sum().item()

                epoch_loss += loss.item() * batch_size
                epoch_total += batch_size
                epoch_correct += (predicted == labels).sum().item()

                batch_count += 1

                if batch_count % 100 == 0:
                    batch_loss = running_loss / 100
                    batch_acc = 100 * running_correct / running_total
                    batch_time = time.time() - start_time

                    print(f'[{epoch + 1}, {batch_count:5d}, step {global_step}] loss: {batch_loss:.3f} acc: {batch_acc:.2f} time: {batch_time:.2f}')

                    if use_wandb:
                        wandb.log({
                            "step": global_step,
                            "batch": epoch * batch_count + batch_count,
                            "batch_loss": batch_loss,
                            "batch_accuracy": batch_acc,
                            "batch_time": batch_time,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })

                    running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                    start_time = time.time()

        if epoch_total > 0:
            epoch_loss = epoch_loss / epoch_total
            epoch_acc = 100 * epoch_correct / epoch_total
        else:
            epoch_loss, epoch_acc = 0.0, 0.0

        net.eval()
        val_acc, val_loss = run_test(net, valloader, criterion, device, task, validation=True)

        accuracy_train.append(epoch_acc)
        loss_train.append(epoch_loss)
        accuracy_val.append(val_acc)
        loss_val.append(val_loss)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "total_steps": global_step
            })

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
              f'Steps: {global_step}')

        print("Salvataggio modello")
        model_filename = f"ROT_ViT_{epoch+1}.pth"
        rotation_model_path = save_model_to_wandb(net, model_filename, run)
        print(f"Model saved as {model_filename}")
        

        if early_stopping:
            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = net.state_dict()
            else:
                epochs_no_improve += 1
                print(f"→ No improvement for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1} (step {global_step})")
                    if best_model_state:
                        net.load_state_dict(best_model_state)
                    break

    print('Finished Training')
    return accuracy_train, loss_train, accuracy_val, loss_val

#this will be our train function after the pre-train
def train_wandb(net, criterion, optimizer, scheduler, num_epochs, device,
                task, trainloader, valloader, early_stopping,
                patience, delta, use_wandb=True):
    accuracy_train = []
    loss_train = []
    accuracy_val = []
    loss_val = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        net.train()
        running_loss, running_correct, running_total = 0.0, 0.0, 0.0
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        start_time = time.time()

        for i, (imgs, imgs_rotated, rotation_label, cls_label) in enumerate(trainloader, 0):
            if task == 'rotation':
                inputs, labels = imgs_rotated.to(device), rotation_label.to(device)
            elif task == 'classification':
                inputs, labels = imgs.to(device), cls_label.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            epoch_loss += loss.item() * labels.size(0)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            if i % 25 == 24:
                batch_loss = running_loss / 25
                batch_acc = 100 * running_correct / running_total
                batch_time = time.time() - start_time

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {batch_loss:.3f} acc: {batch_acc:.2f} time: {batch_time:.2f}')

                if use_wandb:
                    wandb.log({
                        "batch": epoch * len(trainloader) + i,
                        "batch_loss": batch_loss,
                        "batch_accuracy": batch_acc,
                        "batch_time": batch_time
                    })

                running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                start_time = time.time()

        epoch_loss = epoch_loss / epoch_total
        epoch_acc = 100 * epoch_correct / epoch_total

        net.eval()
        val_acc, val_loss = run_test(net, valloader, criterion, device, task, validation=True)

        accuracy_train.append(epoch_acc)
        loss_train.append(epoch_loss)
        accuracy_val.append(val_acc)
        loss_val.append(val_loss)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        scheduler.step()

        if early_stopping:
            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = net.state_dict()
            else:
                epochs_no_improve += 1
                print(f"→ No improvement for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    if best_model_state:
                        net.load_state_dict(best_model_state)
                    break

    print('Finished Training')
    return accuracy_train, loss_train, accuracy_val, loss_val

def check_image_channels(example):
    try:
        if isinstance(example['image'], (list, np.ndarray)):
            img = example['image']
            if len(img.shape) == 2:  # grayscale (H, W)
                print(f"Problem: Image {example['label']} - Shape: {img.shape}")
                return {"is_problematic": True}
            elif len(img.shape) == 3 and img.shape[0] != 3:  # not RGB
                print(f"Problem: Image {example['label']} - Shape: {img.shape}")
                return {"is_problematic": True}
        elif hasattr(example['image'], 'mode'):
            if example['image'].mode != 'RGB':
                print(f"Problem: Image {example['label']} - Mode: {example['image'].mode}")
                return {"is_problematic": True}
        return {"is_problematic": False}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"is_problematic": True}
    
def convert_to_rgb(example):
    if hasattr(example['image'], 'convert'):
        return {
            "image": example['image'].convert("RGB"),
            "label": example['label']
        }
    else:
        return example