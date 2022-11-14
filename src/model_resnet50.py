# model.py

import torch.nn as nn
import pretrainedmodels


def get_model(pretrained, trainable_layer_count="all"):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
    else:
        model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
    print(f"Number of trainable layers -- {trainable_layer_count}")
    if trainable_layer_count == "all":
        # the full pre-trained model is fine-tuned in this case
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True
    else:
        ct = 0
        num_child = len(list(model.children()))
        non_trainable_layer_count = num_child - trainable_layer_count
        if non_trainable_layer_count < 0:
            raise Exception(
                f"number of trainable layer less is {non_trainable_layer_count} which is less than 0, try changing trainable_layer_count "
            )
        for child in model.children():
            ct += 1
            if ct < non_trainable_layer_count:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    print("base model has {} layers".format(len(list(model.children()))))

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=1),
        nn.Sigmoid(),
    )

    return model
