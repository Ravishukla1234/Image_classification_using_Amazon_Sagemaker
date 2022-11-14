# train.py
import os

os.system("pip uninstall opencv-python==4.5.2.52 -y")
os.system("pip install opencv-python-headless==4.5.5.62")
os.system("pip install -U albumentations")

os.system("pip install -q pretrainedmodels")
os.system("pip install -q typing_extensions==4.0.0")

os.system("pip install -q tqdm")


import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import dataset
import engine
from model_resnet50 import get_model
from tqdm import tqdm

import math
import cv2
import argparse
import json
import pickle
from PIL import Image


def get_data(data_dir, labels):
    """
    parse the input images and retrieve their paths and labels

    :param data_dir : directory containing training images
    :param labels   : list of output labels, the data_dir should have folder with images for each unique labels
    :return image path and labels
    """
    print(f"Loading the image dataset")
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in tqdm(os.listdir(path), total=len(os.listdir(path))):
            try:
                img_path = os.path.join(path, img)
                data.append([img_path, class_num])
            except Exception as e:
                print(e)
    X, y = [], []
    for feature, label in data:
        X.append(feature)
        y.append(label)
    print(f"Done")
    return X, y


def _save_model(model, model_dir):
    """
    save pytorch model

    :param model       : pytorch model
    :param model_dir   : directory to save pytorch model
    """
    print("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def _train(args):
    """
    run training

    :param args  : argparse namespace
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device Type: {}".format(device))

    ## Read basic arguments for model training

    epochs = args.epochs
    trainable_layer_count = args.trainable_layer_count
    if trainable_layer_count != "all":
        trainable_layer_count = int(trainable_layer_count)
    lr = args.lr
    data_dir = args.data_dir
    model_dir = args.model_dir

    print(f"Basic Arguments       -- ")
    print(f"epochs                -- {epochs}")
    print(f"trainable_layer_count -- {trainable_layer_count}")
    print(f"lr                    -- {lr}")
    print(f"data_dir              -- {data_dir}")
    print(f"model_dir             -- {model_dir}")
    print(f"epochs                -- {epochs}")

    ## load the image dataset
    labels = ["cat", "dog"]
    images, targets = get_data(data_dir, labels)
    print(f"Unique labels -- {set(targets)}")

    # fetch out model
    model = get_model(pretrained=True, trainable_layer_count=trainable_layer_count)
    # move model to device
    model.to(device)

    # mean and std values of RGB channels for imagenet dataset
    # we use these pre-calculated values when we use weights
    # from imagenet.
    # when we do not use imagenet weights, we use the mean and
    # standard deviation values of the original dataset
    # please note that this is a separate calculation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # albumentations is an image augmentation library
    # that allows you to do many different types of image
    # augmentations. here, i am using only normalization
    # notice always_apply=True. we always want to apply
    # normalization
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )
    # instead of using kfold, i am using train_test_split
    # with a fixed random state
    train_images, valid_images, train_targets, valid_targets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )
    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(227, 227),
        augmentations=aug,
    )
    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )
    # same for validation data
    valid_dataset = dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227, 227),
        augmentations=aug,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    ## Training

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    # train and print accuracy score for all epochs
    best_acc = 0
    best_epoch = -1
    for epoch in range(epochs):
        model.to(device)
        print(f"epoch -- {epoch}")
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(valid_loader, model, device=device)
        predictions = np.array([x[0] for x in predictions])
        preds = np.where(predictions > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(valid_targets, preds)

        print(f"Epoch={epoch}, Valid accuracy = {accuracy}")
        if accuracy > best_acc:
            print("Best epoch until now....")
            best_acc = accuracy
            best_epoch = epoch
            print(f"Saving model to {args.model_dir}")
            _save_model(model, args.model_dir)
    print(f"Best accuracy - {best_acc} at epoch - {best_epoch}")
    print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        metavar="BS",
        help="batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="initial learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="distributed backend (default: gloo)",
    )
    parser.add_argument(
        "--trainable_layer_count",
        type=str,
        default="6",
        metavar="BS",
    )

    parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())


def model_fn(model_dir):
    """
    load and return serialized pytorch model during inference

    :param model_dir   : directory having saved pytorch model
    :return serialized pytorch model
    """
    print("model_fn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(pretrained=True)

    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def model_fn(model_dir):
    """
    load and return serialized pytorch model during inference

    :param model_dir   : directory having saved pytorch model
    :return serialized pytorch model
    """
    print("model_fn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(pretrained=True)

    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def input_fn(serialized_input_data, content_type):
    """
    load serialized_input_data ,preprocess and return it

    :param serialized_input_data  : serialized input coming from endpoint
    :param content_type           : content_type of the image
    :return preprocessed image as pytorch tensor
    """
    data = serialized_input_data

    # use PIL to load the image
    print(f"1 - {np.array(data).shape}")

    n = int(math.sqrt((np.array(data).shape[0]) / 3))

    shape = int(n * n * 3)
    print(shape)

    image = Image.fromarray(np.array(data)[-shape:].reshape(int(n), int(n), 3), "RGB")

    # convert image to RGB, we have single channel images and resize
    image = image.resize((227, 227), Image.BILINEAR).convert("RGB")

    # convert image to numpy array
    image = np.array(image)
    print(f"2 - {np.array(image).shape}")
    # add albumentation augmentations

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    augmentations = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )
    augmented = augmentations(image=image)
    image = augmented["image"]
    print(f"3 - {np.array(image).shape}")

    # pytorch expects CHW instead of HWC
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, 0)

    print(f"Preprocessed image shape -- {image.shape}")
    print(image)
    image = torch.tensor(image, dtype=torch.float)
    return image
