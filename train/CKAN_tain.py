import mydataset_HTRU
from ckan_model.model_convKan import KANC_MLP
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from htru1 import HTRU1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = HTRU1('data', train=True, download=True, transform=transform)
virtue_train_dataset = mydataset_HTRU.myDataset("./htru_dataset/dataset.csv")
combined_train_dataset = ConcatDataset([virtue_train_dataset, train_data])

virtue_test_dataset = mydataset_HTRU.myDataset('./htru_dataset/vir_dataset.csv')
test_data = HTRU1('data', train=False, download=True, transform=transform)
combined_test_dataset = ConcatDataset([virtue_test_dataset, test_data])

train_dataset = mydataset_HTRU.select_sample(combined_train_dataset , true_num=15000 , false_num=30000, true_class=0 , false_class=1)

test_recall = mydataset_HTRU.select_sample(combined_test_dataset , true_num=1000 ,false_num=0 , true_class=0 , false_class=1)



train_data_size = len(train_dataset)
test_data_size = len(test_recall)
print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))


trainloader = DataLoader(train_dataset, batch_size=64 , shuffle=True)
valloader = DataLoader(test_recall, batch_size=64 , shuffle=True)

# Define loss

criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.,1.]).to(device))

model = KANC_MLP(device=device)

model = nn.DataParallel(model).to(device)
# print(model)



# Define optimizer
# * `optim.AdamW`: 这是Adam优化器的一个变种，称为AdamW（或称为Weight Decay in Adam）。它与Adam的主要区别在于，它实现了权重衰减（weight decay）作为Adam优化过程的一部分，而不是作为外部的正则化项。
# * `model.parameters()`: 这返回模型中所有可训练的参数。
# * `lr=1e-3`: 学习率设置为0.001。
# * `weight_decay=1e-4`: 权重衰减设置为0.0001。权重衰减是一种正则化技术，用于防止模型过拟合。
optimizer = optim.AdamW(model.parameters(), lr=0.01)


# Define learning rate scheduler
# * `optim.lr_scheduler.ExponentialLR`: 这是一个学习率调度器，用于在每个epoch后按指数方式调整学习率。
# * `optimizer`: 前面定义的优化器实例。
# * `gamma=0.8`: 这决定了学习率在每个epoch后减少的速率。具体来说，新的学习率将是当前学习率乘以`gamma`。因此，如果初始学习率是0.001，那么在下一个epoch后，学习率将变为0.001 * 0.8 = 0.0008。
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

wandb.watch(model)
for epoch in range(50):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar): #用tqdm进度条可视化训练过程
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
            # wandb.log({'train_loss': loss, 'train_accuracy': accuracy})
    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )

    val_loss /= len(valloader)
    val_accuracy /= len(valloader)
    wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy})
    # Update learning rate
    scheduler.step()
    # if ((epoch + 1) % 10 == 0):
    #     if isinstance(model, torch.nn.DataParallel):
    #         torch.save(model.module.state_dict(), './net_pth2/net_kan_{}.pth'.format(epoch + 1))
    #     else:
    #         torch.save(model.state_dict(), './net_pth2/net_kan_{}.pth'.format(epoch + 1))
    #     print("模型{}已保存".format(epoch + 1))
    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )

wandb.finish()