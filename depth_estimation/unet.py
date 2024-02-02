import numpy as np
import torch
import omegaconf
from depth_estimation.omnidepth.unet.model import UNet as Model
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import cv2


model_urls = {
    'UNet': 'https://github.com/tzole1155/StreamLitDemo/releases/download/Unet/unet.pth',
}


def get_model(device, **kwargs):
    is_submodule = kwargs.get('is_submodule', False)
    if is_submodule:
        conf_path = './pano_da/depth_estimation/omnidepth/unet/model_unet.yaml'
    else:
        conf_path = './depth_estimation/omnidepth/unet/model_unet.yaml'
    conf = omegaconf.OmegaConf.load(conf_path)
    model = Model(conf.model.configuration)

    checkpoint = torch.load('./pretrained_depth_models/unet_release.pth')
    model.load_state_dict(checkpoint.state_dict(), False)
    
    model.to(device)
    model.eval()
    
    return model


def preprocess(image):
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LANCZOS4)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

    return image
