import torch

from colmap.read_write_model import Camera

from ovl.camera.camera_model.camera_model import CameraModel
from ovl.utils import ignore_warnings


class PinholeCameraModel(CameraModel):
    r"""Represents a pinhole camera model.

    Parameters
    ----------
    focal : array-like
        (fx, fy)
    principal : array-like
        (cx, cy)
    size_wh : array-like
        (w, h)
    """
    @ignore_warnings(['To copy construct from a tensor, it is recommended to use'])
    def __init__(self, focal, principal, size_wh):
        super().__init__(size_wh)
        self.focal = torch.nn.Parameter(torch.tensor(focal), requires_grad=False)
        self.principal = torch.nn.Parameter(torch.tensor(principal), requires_grad=False)

    def clone(self):
        return PinholeCameraModel(self.focal.clone(), self.principal.clone(), self.size_wh.clone())

    @classmethod
    def from_colmap(cls, colmap_cam):
        r"""Makes the model from the COLMAP Camera.

        Parameters
        ----------
        colmap_cam : Camera

        Returns
        -------
        cam_model : PinholeCameraModel
        """
        fx, fy, cx, cy = colmap_cam.params
        return cls([fx, fy], [cx, cy], [colmap_cam.width, colmap_cam.height])

    def to_colmap(self, cam_id=0):
        r"""Converts the camera model to the COLMAP Camera.

        Parameters
        ----------
        cam_id : int

        Returns
        -------
        colmap_cal : Camera
        """
        w, h = self.size_wh.cpu().tolist()
        params = torch.cat([self.focal.cpu(), self.principal.cpu()]).numpy()
        return Camera(id=cam_id, model='PINHOLE', width=w, height=h, params=params)

    def __repr__(self):
        return (f'PinholeCameraModel(size_wh={tuple(self.size_wh.tolist())}, '
                f'focal={tuple(self.focal.half().tolist())}, '
                f'principal={tuple(self.principal.half().tolist())})')

    @property
    def dtype(self):
        return self.focal.dtype

    def project(self, xyz):
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
        uv = xyz[:2] / xyz[2:3]; del xyz
        uv = uv.mul_(self.focal.unsqueeze(1)).add_(self.principal.unsqueeze(1))
        return uv

    def unproject(self, uv):
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
        n = uv.shape[1]
        direction = (uv - self.principal.unsqueeze(1)).div_(self.focal.unsqueeze(1))
        direction = torch.cat([direction, direction.new_ones(1, n)], 0)
        direction = torch.nn.functional.normalize(direction, dim=0)
        return direction

    def resize_(self, new_wh):
        r"""Resizes camera model inplace to a new resolution.

        Parameters
        ----------
        new_wh : torch.Tensor
        """
        old_wh = self.size_wh.clone()
        new_wh = new_wh.to(old_wh)
        self.size_wh.copy_(new_wh)
        self.focal.data.mul_(new_wh).div_(old_wh)
        self.principal.data.mul_(new_wh).div_(old_wh)
        return self

    def crop_(self, crop_left_top, new_wh=None):
        self.principal.data[0] -= crop_left_top[0]
        self.principal.data[1] -= crop_left_top[1]
        if new_wh is None:
            self.size_wh.data[0] -= crop_left_top[0]
            self.size_wh.data[1] -= crop_left_top[1]
        else:
            self.size_wh.data[0] = new_wh[0]
            self.size_wh.data[1] = new_wh[1]
        return self

