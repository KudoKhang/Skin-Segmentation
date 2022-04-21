from utils.librarys import *
from utils.funcs import bbox_overlaps, eval_model, process_data
from models.dataloader import Skin_dataset
from models.augment import *
from models.maskrcnn import *

def run():
    # Process the data
    train_label, val_label = tqdm(process_data('datasets/CROP/', 'datasets/Skin_segmentation_final.json'))
    print('Train:', len(train_label))

    # Load data
    dataset = Skin_dataset(train_label, transforms=get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=2, 
                                            shuffle=True,
                                            num_workers=2, 
                                            collate_fn=collate_fn)

    dataset_test = Skin_dataset(val_label, transforms=get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
                                            dataset_test,
                                            batch_size=2,
                                            shuffle=False,
                                            num_workers=2,
                                            collate_fn=collate_fn)

    # Init model
    model = resnet101_maskRCNN(5, True)

    # Load pre-trained weights
    # model.load_state_dict(torch.load('/content/drive/MyDrive/Khanghn/resnet101_skin_segmentation_GIRL_final_epoch.pth'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Training on: ', device)
    model.to(device);

    # Hyperparameter
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-05, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    # Connect wandb
    wandb.init(project="skin_segmentation", entity="khanghn")

    n_batch = len(data_loader)
    max_map = 0
    n_epoch = 300

    for epoch in range(n_epoch):
        model.train()
        losses_record = []
        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for batch_idx, (images, targets) in enumerate(tepoch):

                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # Backward and optimize
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                losses_record.append(losses.item())
                tepoch.set_postfix(loss=losses_record[-1])

                if batch_idx >= n_batch - 1:
                    
                    mm = eval_model(model, data_loader_test)
                    ll = np.mean(losses_record)

                    # Wandb_log
                    wandb.log({"MAP": mm, "Loss": ll})
                    wandb.watch(model)

                    if max_map < mm:
                        max_map = mm

                        torch.save(model.state_dict(), "/weights/resnet101_skin_segmentation_GIRL.pth")
                        tepoch.set_postfix(loss = ll, max_map=mm, save_weight = 'True')
                    else:
                        tepoch.set_postfix(loss = ll, max_map=mm, save_weight = 'False')
                    
                    torch.save(model.state_dict(), "/weights/resnet101_skin_segmentation_GIRL_final_epoch.pth")  

            lr_scheduler.step(ll)

if __name__ == '__main__':
    run()
