import timm
import torch.nn as nn

from torchvision.models import resnet50, efficientnet_b7, convnext_base, vit_b_16, maxvit_t, resnext50_32x4d, \
                                swin_b, swin_v2_b, convnext_large
from torchvision.models.efficientnet import efficientnet_v2_s

class SingleModel(nn.Module):
    def __init__(self, type, num_class) -> None:
        super().__init__()
        if type == 'resnet50':
            self.model = resnet50(weights='DEFAULT')
            self.model.fc = nn.Linear(2048, num_class) # print(model) to get the layer info
        elif type == 'efficientnet_b7':
            self.model = efficientnet_b7(weights='DEFAULT')
            self.model.classifier[1] = nn.Linear(2560, num_class)
        elif type == 'efficientnet_v2_s':                    
            self.model = efficientnet_v2_s(weights='DEFAULT')  
            self.model.classifier[1] = nn.Linear(1280, num_class)
        elif type == 'convnext_base':               
            self.model = convnext_base(weights='DEFAULT')  
            self.model.classifier[2] = nn.Linear(1024, num_class)
        elif type == 'convnext_large':               
            self.model = convnext_large(weights='DEFAULT')  
            self.model.classifier[2] = nn.Linear(1536, num_class)
        elif type == 'vit_b_16':
            self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.model.heads.head = nn.Linear(768, num_class)
        elif type == 'max_vit':
            self.model = timm.create_model('maxvit_tiny_tf_512.in1k', pretrained=True, num_classes=num_class)
        elif type == 'resnext50':
            self.model = resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
            self.model.fc = nn.Linear(2048, num_class)
        elif type == 'swin_b':
            self.model = swin_b(weights='DEFAULT')
            self.model.head = nn.Linear(1024, num_class)
        elif type == 'swin_v2_b':
            self.model = swin_v2_b(weights='DEFAULT')
            self.model.head = nn.Linear(1024, num_class)
    def forward(self, x):
        return self.model(x)

class SingleModelCombineDataset(nn.Module):
    def __init__(self, type) -> None:
        super().__init__()
        if type == 'resnet50':
            self.model = resnet50(weights='DEFAULT')
            self.model.fc = nn.Linear(2048, 19) 
        elif type == 'efficientnet_b7':
            self.model = efficientnet_b7(weights='DEFAULT')
            self.model.classifier[1] = nn.Identity()
            self.fc_colon = nn.Linear(2560, 4)
            self.fc_gastric = nn.Linear(2560, 4)
            self.fc_prostate = nn.Linear(2560, 4)
            self.fc_k19 = nn.Linear(2560, 9)
        elif type == 'efficientnet_v2_s':                    
            self.model = efficientnet_v2_s(weights='DEFAULT')  
            self.model.classifier[1] = nn.Identity()
            self.fc_colon = nn.Linear(1280, 4)
            self.fc_gastric = nn.Linear(1280, 4)
            self.fc_prostate = nn.Linear(1280, 4)
            self.fc_k19 = nn.Linear(1280, 9)
        elif type == 'convnext_base':               
            self.model = convnext_base(weights='DEFAULT')  
            self.model.classifier[2] = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)
        elif type == 'convnext_large':               
            self.model = convnext_large(weights='DEFAULT')  
            self.model.classifier[2] = nn.Identity()
            self.fc_colon = nn.Linear(1536, 4)
            self.fc_gastric = nn.Linear(1536, 4)
            self.fc_prostate = nn.Linear(1536, 4)
            self.fc_k19 = nn.Linear(1536, 9)
        elif type == 'vit_b_16':
            self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.model.heads.head = nn.Identity()
            self.fc_colon = nn.Linear(768, 4)
            self.fc_gastric = nn.Linear(768, 4)
            self.fc_prostate = nn.Linear(768, 4)
            self.fc_k19 = nn.Linear(768, 9)
        # elif type == 'max_vit':
        #     self.model = timm.create_model('maxvit_tiny_tf_512.in1k', pretrained=True, num_classes=num_class)
        elif type == 'resnext50':
            self.model = resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
            self.model.fc = nn.Identity()
            self.fc_colon = nn.Linear(2048, 4)
            self.fc_gastric = nn.Linear(2048, 4)
            self.fc_prostate = nn.Linear(2048, 4)
            self.fc_k19 = nn.Linear(2048, 9)
        elif type == 'swin_b':
            self.model = swin_b(weights='DEFAULT')
            self.model.head = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)
        elif type == 'swin_v2_b':
            self.model = swin_v2_b(weights='DEFAULT')
            self.model.head = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)

    def forward(self, x):
        x_main = self.model(x)
        # x_colon = self.fc_colon_2(self.fc_colon(x_main))
        # x_gastric = self.fc_gastric_2(self.fc_gastric(x_main))
        # x_prostate = self.fc_prostate_2(self.fc_prostate(x_main))
        # x_k19 = self.fc_k19_2(self.fc_k19(x_main))
        return x_main

class SingleModelMultiBranch(nn.Module):
    """
    colon: 4
    gastric: 4
    prostate: 4
    k19: 9
    """

    def __init__(self, type) -> None:
        super().__init__()
        
        if type == 'resnet50':
            self.model = resnet50(weights='DEFAULT')
            self.model.fc = nn.Identity() # print(model) to get the layer info
            self.fc_colon = nn.Linear(2048, 4)
            #self.fc_colon_2 = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(2048, 4)
            #self.fc_prostate_2 = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(2048, 4)
            #self.fc_gastric_2 = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(2048, 9)
            #self.fc_k19_2 = nn.Linear(1024, 9)
        elif type == 'efficientnet_b7':
            self.model = efficientnet_b7(weights='DEFAULT')
            self.model.classifier[1] = nn.Identity()
            self.fc_colon = nn.Linear(2560, 4)
            self.fc_gastric = nn.Linear(2560, 4)
            self.fc_prostate = nn.Linear(2560, 4)
            self.fc_k19 = nn.Linear(2560, 9)
        elif type == 'efficientnet_v2_s':                    
            self.model = efficientnet_v2_s(weights='DEFAULT')  
            self.model.classifier[1] = nn.Identity()
            self.fc_colon = nn.Linear(1280, 4)
            self.fc_gastric = nn.Linear(1280, 4)
            self.fc_prostate = nn.Linear(1280, 4)
            self.fc_k19 = nn.Linear(1280, 9)
        elif type == 'convnext_base':               
            self.model = convnext_base(weights='DEFAULT')  
            self.model.classifier[2] = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)
        elif type == 'convnext_large':               
            self.model = convnext_large(weights='DEFAULT')  
            self.model.classifier[2] = nn.Identity()
            self.fc_colon = nn.Linear(1536, 4)
            self.fc_gastric = nn.Linear(1536, 4)
            self.fc_prostate = nn.Linear(1536, 4)
            self.fc_k19 = nn.Linear(1536, 9)
        elif type == 'vit_b_16':
            self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.model.heads.head = nn.Identity()
            self.fc_colon = nn.Linear(768, 4)
            self.fc_gastric = nn.Linear(768, 4)
            self.fc_prostate = nn.Linear(768, 4)
            self.fc_k19 = nn.Linear(768, 9)
        # elif type == 'max_vit':
        #     self.model = timm.create_model('maxvit_tiny_tf_512.in1k', pretrained=True, num_classes=num_class)
        elif type == 'resnext50':
            self.model = resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
            self.model.fc = nn.Identity()
            self.fc_colon = nn.Linear(2048, 4)
            self.fc_gastric = nn.Linear(2048, 4)
            self.fc_prostate = nn.Linear(2048, 4)
            self.fc_k19 = nn.Linear(2048, 9)
        elif type == 'swin_b':
            self.model = swin_b(weights='DEFAULT')
            self.model.head = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)
        elif type == 'swin_v2_b':
            self.model = swin_v2_b(weights='DEFAULT')
            self.model.head = nn.Identity()
            self.fc_colon = nn.Linear(1024, 4)
            self.fc_gastric = nn.Linear(1024, 4)
            self.fc_prostate = nn.Linear(1024, 4)
            self.fc_k19 = nn.Linear(1024, 9)

    def forward(self, x):
        x = self.model(x)
        # x_colon = self.fc_colon_2(self.fc_colon(x))
        # x_gastric = self.fc_gastric_2(self.fc_gastric(x))
        # x_prostate = self.fc_prostate_2(self.fc_prostate(x))
        # x_k19 = self.fc_k19_2(self.fc_k19(x))

        x_colon = self.fc_colon(x)
        x_gastric = self.fc_gastric(x)
        x_prostate = self.fc_prostate(x)
        x_k19 = self.fc_k19(x)
        return x_colon, x_gastric, x_prostate, x_k19