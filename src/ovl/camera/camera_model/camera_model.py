from abc import ABC, abstractmethod

import torch

from ovl.utils import ignore_warnings


class CameraModel(ABC, torch.nn.Module):
    r"""Represents an abstract camera model.

    Parameters
    ----------
    size_wh : tuple of int
        Image dimensions.
    """
    @ignore_warnings(['To copy construct from a tensor, it is recommended to use'])
    def __init__(self, size_wh):
        super().__init__()
        size_wh = torch.tensor(size_wh)
        if torch.is_floating_point(size_wh):
            raise ValueError(f'Expected integer size_wh, got {size_wh}')
        self.register_buffer('size_wh', size_wh)

    @property
    def device(self):
        return self.size_wh.device

    @property
    @abstractmethod
    def dtype(self): ...

    @abstractmethod
    def project(self, xyz, **kwargs):
        r"""Projects the cam-space points to the image space.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n].

        Returns
        -------
        uv : torch.Tensor
            of shape [2, n]. Image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        ...

    @abstractmethod
    def unproject(self, uv, **kwargs):
        r"""Calculates cam-space 3D directions corresponding to the image-space points.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n]. Image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).

        Returns
        -------
        direction : torch.Tensor
            of shape [3, n]. Directions in cam space, X to the right, Y down, Z from the camera.
        """
        ...

    @abstractmethod
    def resize_(self, new_wh):
        r"""Resizes camera model inplace to a new resolution.

        Parameters
        ----------
        new_wh : torch.Tensor
        """
        ...

    def get_pix_rays(self, uv_shift=(0, 0)):
        r"""Calculates the cam-space 3D directions corresponding to pixel centers.

        Parameters
        ----------
        uv_shift : tuple of float
            Shift of the centers in image space.

        Returns
        -------
        direction : torch.Tensor
            of shape [3, h, w]. Directions in cam space, X to the right, Y down, Z from the camera, NaN if not calibrated.
        """
        w, h = self.size_wh.cpu().tolist()
        uv = self.get_pix_uvs(uv_shift)
        directions = self.unproject(uv.view(2, -1)); del uv
        directions = directions.view(3, h, w)
        return directions

    def get_pix_uvs(self, uv_shift=(0, 0)):
        r"""Calculates UV coordinates of pixel centers.

        Parameters
        ----------
        uv_shift : tuple of float
            Shift of the centers in image space.

        Returns
        -------
        uv : torch.Tensor
            of shape [2, h, w]. Image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).
        """
        w, h = self.size_wh.cpu().tolist()
        u_shift, v_shift = uv_shift
        v, u = torch.meshgrid([torch.linspace(.5 + v_shift, h - .5 + v_shift, h, device=self.device, dtype=self.dtype),
                               torch.linspace(.5 + u_shift, w - .5 + u_shift, w, device=self.device, dtype=self.dtype)],
                              indexing='ij')
        uv = torch.stack([u, v]); del u, v
        return uv

    def get_pix_ids(self, xyz):
        r"""Projects the cam-space points to the image space, and computes the respective pixel ids.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n].

        Returns
        -------
        pix_i : torch.Tensor
            of shape [points_n]. Index of the pixel where the point is projected to.
        """
        uv = self.project(xyz)
        is_in_bounds = self.uv_is_in_bounds(uv)
        u, v = uv; del uv
        w, h = self.size_wh
        j = torch.empty_like(u, dtype=torch.long).copy_(u); del u  # u, v are floored here
        i = torch.empty_like(v, dtype=torch.long).copy_(v); del v
        pix_i = i.mul_(w).add_(j); del i, j
        pix_i = pix_i.where(is_in_bounds, pix_i.new_tensor(-1))
        return pix_i

    def uv_is_in_bounds(self, uv):
        r"""Checks if the image-space points are within the canvas.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [2, n]. Image coordinates of the points.
            The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (0,0), the coordinates of the bottom-right corner are (w,h).

        Returns
        -------
        is_in_bounds : torch.BoolTensor
            of shape [n].
        """
        u, v = uv
        w, h = self.size_wh
        is_in_bounds = (u < w).logical_and_(u >= 0).logical_and_(v < h).logical_and_(v >= 0)
        return is_in_bounds

    def uv_to_torch(self, uv):
        r"""Converts UV coordinates to torch format.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [n, 2]. Transposed UV coordinates from `project`.

        Returns
        -------
        uv_torch : torch.Tensor
            of shape [n, 2]. The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (-1,-1), the coordinates of the bottom-right corner are (1,1).
        """
        uv = (uv / (self.size_wh.to(uv) / 2)).sub_(1)
        return uv
