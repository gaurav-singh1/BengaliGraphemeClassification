import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliaiDataset
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE='cpu'
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

BASE_MODEL = os.environ.get("BASE_MODEL")

def loss_fn(output, targets):
    o1, o2, o3 = output
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)

    return (l1 + l2 + l3)/3





def train(model, data_loader, optimizer):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for d in tk0:
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        consonant_diacritic = d["consonant_diacritic"]
        vowel_diacritic = d["vowel_diacritic"]

        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)

        optimizer.zero_grad()

        output = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        
def evaluate(model, data_loader):
    model.eval()
    
    tk1 = tqdm(data_loader, total=len(data_loader))
    final_loss = 0
    for d in tk1:
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        consonant_diacritic = d["consonant_diacritic"]
        vowel_diacritic = d["vowel_diacritic"]

        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)

        with torch.no_grad():
            output = model(image)
            targets = (grapheme_root, consonant_diacritic, vowel_diacritic)
            loss = loss_fn(output, targets)
            final_loss+=loss.item()
    
    return final_loss/len(data_loader)



def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliaiDataset(folds=TRAINING_FOLDS,
                                    img_height=IMG_HEIGHT,
                                    img_width=IMG_WIDTH,
                                    mean=MODEL_MEAN,
                                    std=MODEL_STD)

    training_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4)


    valid_dataset = BengaliaiDataset(folds=VALIDATION_FOLDS,
                                    img_height=IMG_HEIGHT,
                                    img_width=IMG_WIDTH,
                                    mean=MODEL_MEAN,
                                    std=MODEL_STD)

    validation_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optimizer, 
                                                            mode='max', 
                                                            factor=0.3, 
                                                            patience=5, 
                                                            verbose=True)


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel()
    
    for epoch in range(EPOCHS):
        train(model, training_loader, optimizer)
        val_score = evaluate(model, validation_loader)
        scheduler.step(val_score)
        torch.save(model.state_dict(), "{}_fold{}.bin".format(BASE_MODEL, VALIDATION_FOLDS[0]))
    


if __name__ == "__main__":
    main()



    

