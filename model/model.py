import torch
import torch.nn as nn
import clip
import os

from resnet.resnet import resnet50
from transformers import AutoTokenizer, OPTForCausalLM, CLIPVisionModelWithProjection
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
from torchvision.models import convnext_base, convnext_large, convnext_small
from torchvision.models import vit_b_16
from transformers import ConvNextV2Model
import torch.backends.cudnn as cudnn
from torch.distributed import init_process_group
from transformers.utils import logging
logging.set_verbosity_error()

class MLP(nn.Module):
    def forward(self, x):
        if x.shape[-1] == 1: # efficientnet: bs x 1280 x 1 x 1
            x = x.squeeze((-1, -2))
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class Encoder(nn.Module):
    def forward(self, x):
        return self.model(x)
    
    def forward_sequence(self, x):
        return self.model.forward_sequence(x)

    def __init__(self, type='resnet50'):
        super(Encoder, self).__init__()
        if type == 'resnet50':      
            self.model = resnet50()                            # dim: 2048
            self.img_embed_dim = 2048
        elif type == 'efficientnet_b0':
            self.model = efficientnet_b0(weights='DEFAULT')    # dim: 1280
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1280
        elif type == 'efficientnet_b1':
            self.model = efficientnet_b1(weights='DEFAULT')    # dim: 1280
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1280
        elif type == 'efficientnet_b2':
            self.model = efficientnet_b2(weights='DEFAULT')    # dim: 1408
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1408
        elif type == 'efficientnet_v2_s':                    
            self.model = efficientnet_v2_s(weights='DEFAULT')  # dim: 1280
            layers = list(self.model.children())[:-1]  # layers = list(model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1280
        elif type == 'efficientnet_v2_l':
            self.model = efficientnet_v2_l(weights='DEFAULT')  # dim: 1280
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1280
        elif type == 'efficientnet_v2_m':
            self.model = efficientnet_v2_m(weights='DEFAULT')  # dim: 1280
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1280
        elif type == 'convnext_small':               
            self.model = convnext_small(weights='DEFAULT')      # dim: 768
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 768
        elif type == 'convnext_base':               
            self.model = convnext_base(weights='DEFAULT')      # dim: 1024
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1024
        elif type == 'convnext_large':
            self.model = convnext_large(weights='DEFAULT')     # dim: 1536
            layers = list(self.model.children())[:-1]
            self.model = nn.Sequential(*layers)
            self.img_embed_dim = 1536
        elif type == 'convnext_large_v2':
            self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-large-1k-224")     # dim: 1536
            self.img_embed_dim = 1536
        elif type == 'vit_b_16':
            self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')     # dim: 768
            layers = list(self.model.children())[:-1]
            self.model.heads.head = nn.Linear(768, 768, bias=False)
            self.img_embed_dim = 768
        elif type == 'clip':
            
            self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            self.img_embed_dim = 768
        else:
            raise ValueError('not support this encoder')


class LanguageModel(nn.Module):
    def forward(self, inputs_embeds, attention_mask):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    def generate(self, input_ids, max_length):
        return self.model.generate(input_ids, max_length=max_length)

    def __init__(self, type, freeze):
        super(LanguageModel, self).__init__()
        self.model = OPTForCausalLM.from_pretrained(type)
        self.tokenizer = AutoTokenizer.from_pretrained(type, use_fast=False)
        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
    
    def get_token_embeddings(self):
        return self.model.get_input_embeddings()
    
    
class ImageCaptionModel(nn.Module):
    def __init__(self, args):
        super(ImageCaptionModel, self).__init__()        
        self.encoder = Encoder(type=args.encoder).to(args.device)
        self.mlp = MLP((self.encoder.img_embed_dim, 
                            (args.embedding_size * args.prefix_length) // 2,
                            args.embedding_size * args.prefix_length)).to(args.device)
        self.lm = LanguageModel(type=args.lm, freeze=args.freeze_lm).to(args.device)
        self.args = args

    def init_ddp_env(self, ngpus_per_node):
        self.args.ngpus_per_node = ngpus_per_node
        self.args.local_rank = self.args.device
        torch.cuda.set_device(self.args.device)
        cudnn.benchmark = True
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        init_process_group(
            backend="nccl", 
            init_method='tcp://127.0.0.1:23486',
            world_size=self.args.world_size, 
            rank=torch.cuda.current_device()
        )

    def forward(self, image_tensor, tokens, mask):
        args = self.args
        img_embed = self.encoder(image_tensor)  # Resnet-50: bs x 2048
        if self.args.encoder == 'clip':
            img_embed = img_embed.image_embeds
        elif self.args.encoder == 'convnext_large_v2':
            img_embed = img_embed.pooler_output
        token_embeddings = self.lm.get_token_embeddings().to(args.device)
        embedding_text = token_embeddings(tokens).to(args.device)

        proj_img_embed = self.mlp(img_embed).view(-1, args.prefix_length, args.embedding_size) # bs x prefix_len x embed_dim
        embedding_cat = torch.cat((proj_img_embed, embedding_text), dim=1).to(args.device)     # bs x prefix_len + token_len x embed_dim

        out = self.lm(inputs_embeds=embedding_cat, attention_mask=mask)

        return out

    def get_tokenizer(self):
        return self.lm.tokenizer

