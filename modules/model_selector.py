import torch
from modules.Conformer import Conformer
from modules.LightCNN import LightCNN
from modules.CNNandLSTM import CNNandLSTM
from modules.TMCmodule import TMCmodule
from modules.mmSCG import mmSCG
from modules.Resnet18 import Resnet18
from modules.TransformerModel import TransformerModel
from modules.swinTransformer import SwinTransformer1D
from modules.swinTransformerFPN import SwinTransformerFPN


def select_model(model_name, point_nums, device, **kwargs):
    if model_name == 'Conformer':
        model = Conformer(
            input_channels=9,
            output_channels=4,
            input_time_steps=640,
            output_time_steps=100,
            d_model=256,
            num_layers=4,
            num_heads=8,
            kernel_size=31,
            d_ff=1024,
            dropout=0.1
        )
    elif model_name == 'LightCNN':
        model = LightCNN()
    elif model_name == 'CNNandLSTM':
        model = CNNandLSTM()
    elif model_name == 'TMCmodule':
        model = TMCmodule(device)
    elif model_name == 'mmSCG':
        model = mmSCG(point_nums)
    elif model_name == 'Resnet18':
        model = Resnet18(point_nums)
    elif model_name == 'TransformerModel':
        model = TransformerModel(input_dim=9)
    elif model_name == 'SwinTransformer1D':
        model = SwinTransformer1D(
            num_classes=4,
            in_chans=point_nums,
            embed_dim=128,
            depths=[2, 2, 6],
            num_heads=[4, 8, 16],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.1,
            drop_path_rate=0.2
        )
    elif model_name == 'SwinTransformerFPN':
        model = SwinTransformerFPN(
            num_classes=4,
            in_chans=point_nums,
            embed_dim=128,
            depths=[2, 2, 6],
            num_heads=[4, 8, 16],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.1,
            drop_path_rate=0.2
        )
    else:
        raise ValueError(f"error model: {model_name}")
    
    return model.to(device)
