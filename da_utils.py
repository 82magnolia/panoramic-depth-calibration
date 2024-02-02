import torch
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from pylsd import lsd
import itertools


# Utility fuctions excerpted from https://github.com/sunset1995/HorizonNet
def pano_stretch(img, kx, ky, order=1):
    '''
    img:     [H, W, C]
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    def uv_meshgrid(w, h):
        uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
        uv = uv.astype(np.float64)
        uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
        uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
        return uv

    def _uv_tri(w, h):
        uv = uv_meshgrid(w, h)
        sin_u = np.sin(uv[..., 0])
        cos_u = np.cos(uv[..., 0])
        tan_v = np.tan(uv[..., 1])
        return sin_u, cos_u, tan_v

    def uv_tri(w, h):
        sin_u, cos_u, tan_v = _uv_tri(w, h)
        return sin_u.copy(), cos_u.copy(), tan_v.copy()

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    return stretched_img
