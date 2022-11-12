import cv2
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from GeoNet import *
from d2l import torch as d2l


def train(image_datasets, data_loaders, epochs, learning_rate, wt_decay):
    train_data_size = len(image_datasets['train'])
    test_data_size = len(image_datasets['test'])
    print(train_data_size,test_data_size)

    train_dataloader = data_loaders['train']
    test_dataloader = data_loaders['test']

    # 实例化网络模型
    model = GeoNet()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0

    # 添加tensorboard
    writer = SummaryWriter("logs_train3")

    for i in range(epochs):
        print("----------第{}轮训练开始了---------".format(i + 1))
        model.train()
        total_train_loss = 0
        # 训练步骤开始
        for data in train_dataloader:
            images, targets = data
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_train_loss += loss.item()
            total_train_step += 1

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_train_step % 5 == 0:
                print("训练次数:{},loss:{}".format(total_train_step, loss.item()))

        writer.add_scalar("train_loss", total_train_loss, i+1)

        # 测试步骤开始
        model.eval()
        total_train_accuracy = 0
        total_test_accuracy = 0
        total_test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                images1, targets1 = data
                outputs = model(images1)
                loss = loss_fn(outputs, targets1)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets1.argmax(1)).sum()
                total_test_accuracy += accuracy
            print("在{}轮训练后，整体测试集合上的accuracy:{}".format(i+1, total_test_accuracy / test_data_size))
            writer.add_scalar("test_accuracy", total_test_accuracy / test_data_size, i+1)
            writer.add_scalar("test_loss", total_test_loss, i + 1)
            for data in train_dataloader:
                images2, targets2 = data
                outputs = model(images2)
                accuracy = (outputs.argmax(1) == targets2.argmax(1)).sum()
                total_train_accuracy += accuracy
            print("在{}轮训练后，整体训练集合上的accuracy:{}".format(i+1, total_train_accuracy / train_data_size))
            writer.add_scalar("train_accuracy", total_train_accuracy / train_data_size, i+1)
    writer.close()
    return model

def main():

    MyoPS_C0_split_dir = 'datasets\MyoPS\C0_split'
    MyoPS_LGE_split_dir = 'datasets\MyoPS\LGE_split'
    MyoPS_T2_split_dir = 'datasets\MyoPS\T2_split'

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            #transforms.Resize((256,256))
            transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2))
        ])
    }
    image_datasets = {
        x: MyData(os.path.join(MyoPS_T2_split_dir, x), data_transforms[x])
        for x in ['train', 'test']
    }

    data_loaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0,drop_last=True)
        for x in ['train', 'test']
    }

    model = train(image_datasets, data_loaders, epochs=32, learning_rate=0.01, wt_decay=0)
    torch.save(model.state_dict(), "GeoNet_MyoPS_T2.pth")

if __name__ == '__main__':
    main()
