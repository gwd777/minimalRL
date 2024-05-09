import torch
import numpy as np

_DATASETS = {
    "mvtec": ["data.mvtec", "MVTecDataset"],
    # "hwhq": ["data.hwhq", "HWHQDataset"]
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 将Dataset转换为ndarray
def transDataset2Ndarray(train_dataset):
    x = []
    index_anomaly = []
    index_normal = []
    for index, data_item in enumerate(train_dataset):
        img = data_item["image"]
        is_anomaly = data_item['is_anomaly']
        if is_anomaly:
            index_anomaly.append(index)
        else:
            index_normal.append(index)
        x.append((is_anomaly, img.numpy(), index))  # 将torch.Tensor转换为numpy.ndarray

    # x = np.array(x, dtype=float)
    index_anomaly = np.array(index_anomaly, dtype=int)
    index_normal = np.array(index_normal, dtype=float)
    return index_anomaly, index_normal, x


def transDataset2Tensor(train_dataset):
    x = []
    index_anomaly = []
    index_normal = []
    for index, data_item in enumerate(train_dataset):
        img = data_item["image"]
        img = img.unsqueeze(0)
        img = img.to(device)
        is_anomaly = data_item['is_anomaly']
        if is_anomaly:
            index_anomaly.append(index)
        else:
            index_normal.append(index)
        x.append((is_anomaly, img, index))  # 将torch.Tensor转换为numpy.ndarray

    index_anomaly = np.array(index_anomaly, dtype=int)
    index_normal = np.array(index_normal, dtype=float)
    return index_anomaly, index_normal, x


def get_dataloaders(
        name="mvtec",
        data_path='D:\mvtec_anomaly_detection',
        subdataset='drl_hw_ls_data',
        train_val_split=1.0,
        resize=128,
        imagesize=250,
        rotate_degrees=0,
        translate=0,
        scale=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        gray=0.0,
        hflip=0.0,
        vflip=0.0,
        augment=True,
        seed=123
):

    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    train_dataset = dataset_library.__dict__[dataset_info[1]](
        data_path,
        classname=subdataset,
        resize=resize,
        train_val_split=train_val_split,
        imagesize=imagesize,
        split=dataset_library.DatasetSplit.TRAIN,
        seed=seed,
        rotate_degrees=rotate_degrees,
        translate=translate,
        brightness_factor=brightness,
        contrast_factor=contrast,
        saturation_factor=saturation,
        gray_p=gray,
        h_flip_p=hflip,
        v_flip_p=vflip,
        scale=scale,
        augment=augment,
    )

    test_dataset = dataset_library.__dict__[dataset_info[1]](
        data_path,
        classname=subdataset,
        resize=resize,
        imagesize=imagesize,
        split=dataset_library.DatasetSplit.TEST,
        seed=seed,
    )
    print(f"_________>Dataset: train={len(train_dataset)} test={len(test_dataset)}")
    return train_dataset, test_dataset


if __name__ == '__main__':
    device = 'cuda:0'
    train_dataset, test_dataset = get_dataloaders(
        name="mvtec",
        data_path='D:\mvtec_anomaly_detection',
        subdataset='hw_hq_data', resize=128)    # 'hw_ls_data', 'screw'

    for index, data_item in enumerate(train_dataset):
        img = data_item["image"]
        label = data_item['is_anomaly']
        img = img.to(torch.float).to(device)
        print(index, img.shape, label)

