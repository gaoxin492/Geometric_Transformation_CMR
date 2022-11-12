from torchvision.transforms import transforms
from dataloader import *
from GeoNet import *

def predict(model):
    model.eval()
    total_LGE_accuracy = 0
    total_C0_accuracy = 0

    data_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2))
        ])

    image_datasets_LGE = MyData(os.path.join('datasets\MyoPS\LGE_split','train'), data_aug)+MyData(os.path.join('datasets\MyoPS\LGE_split','test'), data_aug)
    image_datasets_C0 = MyData(os.path.join('datasets\MyoPS\C0_split', 'train'), data_aug) + MyData(os.path.join('datasets\MyoPS\C0_split', 'test'), data_aug)
    data_loaders_LGE =  torch.utils.data.DataLoader(image_datasets_LGE, batch_size=16, shuffle=True, num_workers=0,drop_last=True)
    data_loaders_C0 = torch.utils.data.DataLoader(image_datasets_C0, batch_size=16, shuffle=True, num_workers=0,drop_last=True)

    with torch.no_grad():
        for data in data_loaders_LGE:
            images1, targets1 = data
            outputs = model(images1)
            accuracy = (outputs.argmax(1) == targets1.argmax(1)).sum()
            total_LGE_accuracy += accuracy

        for data in data_loaders_C0:
            images2, targets2 = data
            outputs = model(images2)
            accuracy = (outputs.argmax(1) == targets2.argmax(1)).sum()
            total_C0_accuracy += accuracy
    return total_LGE_accuracy,total_C0_accuracy

if __name__ == '__main__':
    model = GeoNet()  # 要先创建模型框架，再加载之前的状态参数
    model.load_state_dict(torch.load("GeoNet_MyoPS_T2.pth"))
    total_LGE_accuracy,total_C0_accuracy = predict(model)
    print("LGE:{},C0:{}".format(total_LGE_accuracy/1392,total_C0_accuracy/1392))