import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import MatNet
from model import BasicBlock
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    DiceLoss,
)


LEARNING_RATE = 1e-3
DEVICE = "cuda"
BATCH_SIZE = 7
NUM_EPOCHS = 10
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "dataset/images/"
TRAIN_MASK_DIR = "dataset/masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"



def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        targets = targets.cuda().float()
        data = data.cuda().float()
        # forward
        with torch.cuda.amp.autocast():
			
            predictions = model(data).float()
            loss = loss_fn(predictions, targets)
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        


        
def main():
    train_transform = A.Compose([
		A.Normalize (mean=(0, 0, 0), std=(1, 1, 1), p=1.0),
		A.HorizontalFlip(p=0.5),
		A.RandomBrightnessContrast(p=0.3),
		])

    val_transforms = A.Compose([
		A.Normalize (mean=(0, 0, 0), std=(1, 1, 1), p=1.0),
		A.HorizontalFlip(p=0.5),
		])

    model = MatNet(BasicBlock).cuda().float()
    loss_fn = DiceLoss().cuda().float()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("weights/checkpoint-10.pth"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print("Running epoch {}".format(epoch+1))
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, 'weights/checkpoint-{}.pth'.format(epoch+1))

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

main()
