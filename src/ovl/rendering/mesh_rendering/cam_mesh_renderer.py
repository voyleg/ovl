from functools import cached_property

import open3d as o3d
import torch

from ovl.utils import CachedProps
from ovl.camera.camera_model import CameraModel
from ovl.rendering.mesh_rendering.mesh_renderer import MeshRenderer


class CamMeshRenderer(MeshRenderer):
    r"""Renderer for a triangle mesh viewed from a camera, based on Intel Embree ray-tracer.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The mesh to render.

    Attributes
    ----------
    c_rays : torch.Tensor
        of shape [rays_n, 3]. Cam-space directions of optical rays passing through centers of the pixels.
    c2w_rot : torch.Tensor
        of shape [3, 3]. Camera-to-world rotation matrix.
    mesh : o3d.t.geometry.TriangleMesh
        The mesh being rendered.
    pix_ids : tuple of torch.LongTensor
        (i, j) each of shapes [rays_n]. Ids of the pixels corresponding to selected rays, or None if all rays are selected.
    size_wh : tuple of int
        (w, h). Image dimensions.
    tris : torch.IntTensor
        of shape [tris_n, 3]. Vertex index triplets that form the triangles.
    w_rays : torch.Tensor
        of shape [rays_n, 6]. World-space optical rays passing through centers of the pixels,
        each one is (origin_x, _y, _z, direction_x, _y, _z).
    """
    def __init__(self, mesh):
        super().__init__(mesh)
        self.c_rays, self.pix_ids, self.size_wh = None, None, None  # set_cam_model
        self.c2w_rot, self.w_rays = None, None  # set_cam_pose

    def set_cam_model(self, cam_model):
        r"""Sets up cam-space directions of optical rays passing through centers of the pixels.

        Parameters
        ----------
        cam_model : CameraModel
        """
        rays = cam_model.get_pix_rays()
        rays = rays.permute(1, 2, 0).to(self.device, self.dtype)
        h, w = rays.shape[:2]
        self.size_wh = w, h

        pix_ids = rays.isfinite().all(2).nonzero(as_tuple=True)
        if len(pix_ids[0]) == h * w:
            self.pix_ids = None
            self.c_rays = rays.reshape(-1, 3).contiguous()
        else:
            self.pix_ids = pix_ids
            self.c_rays = rays[pix_ids].contiguous()

    def set_cam_pose(self, c2w_t, c2w_rot):
        r"""Sets up world-space rays from the camera pose.

        Parameters
        ----------
        c2w_t : torch.Tensor
            of shape [3]. Camera-to-world translation vector.
        c2w_rot : torch.Tensor
            of shape [3, 3]. Camera-to-world rotation matrix.
        """
        self.w_rays = self.make_rays_from_cam(self.c_rays, c2w_t, c2w_rot)
        self.c2w_rot = c2w_rot

    @staticmethod
    def make_rays_from_cam(c_rays, c2w_t, c2w_rot):
        r"""Transforms optical rays to the world space.

        Parameters
        ----------
        c_rays : torch.Tensor
            of shape [rays_n, 3]. Cam-space directions of optical rays passing through centers of the pixels.
        c2w_t : torch.Tensor
            of shape [3]. Camera-to-world translation vector.
        c2w_rot : torch.Tensor
            of shape [3, 3]. Camera-to-world rotation matrix.

        Returns
        -------
        w_rays : torch.Tensor
            of shape [rays_n, 6]. World-space optical rays, each one is (origin_x, _y, _z, direction_x, _y, _z).
        """
        w_rays = c_rays.new_empty([len(c_rays), 6])
        origins = w_rays[:, :3]
        origins.copy_(c2w_t); del origins, c2w_t
        dirs = w_rays[:, 3:6]
        torch.mm(c_rays, c2w_rot.T, out=dirs); del c2w_rot, dirs
        return w_rays

    def render_pixels(self, attrs=None, cull_backface=False):
        r"""Renders the mesh to the image space.

        Parameters
        ----------
        attrs : iterable of torch.Tensor
            each of shape [verts_n, attrs_n]. Vertex attributes to interpolate at hit points.
        cull_backface : bool
            If True, render the back of the faces as empty.

        Returns
        -------
        render : Render
        """
        ray_render = self.render_rays(self.w_rays, cull_backface)
        ray_render = RayRender(self, ray_render.hit_depth, ray_render.normals, ray_render.tri_ids, ray_render.uvs)
        render = Render(self, ray_render)
        if attrs is not None:
            ray_samples = self.interpolate(attrs, ray_render.tri_ids, ray_render.uvs)
            render.attr_samples = [self.rays_to_pixels(samples) for samples in ray_samples]; del ray_samples
        del ray_render
        return render

    def rays_to_pixels(self, ray_data, default_val=float('nan')):
        r"""Scatters ray data into image pixels.

        Parameters
        ----------
        ray_data : torch.Tensor
            of shape [rays_n, **].
        default_val : float

        Returns
        -------
        pix_data : torch.Tensor
            of shape [height, width, **].
        """
        w, h = self.size_wh
        data_shape = ray_data.shape[1:]
        if self.pix_ids is None:
            pix_data = ray_data.reshape(h, w, *data_shape); del ray_data, data_shape
        else:
            pix_data = ray_data.new_full([h, w, *data_shape], default_val); del data_shape
            pix_data[self.pix_ids] = ray_data; del ray_data
        return pix_data


class RayRender(CachedProps):
    r"""Represents a ray-traced values.

    Parameters
    ----------
    renderer : MeshRenderer

    Attributes
    ---------
    depth_ray, depth_z: torch.Tensor
        of shape [rays_n]. Depth, inf if didn't hit any face.
    is_frontfacing: torch.BoolTensor
        of shape [rays_n]. True for faces oriented towards the camera, False if didn't hit any face.
    normals_world, normals_cam: torch.Tensor
        of shape [rays_n, 3]. Normals, 0 if didn't hit any face.
    ray_dirs_world, ray_dirs_cam: torch.Tensor
        of shape [rays_n, 3]. Ray directions.
    tri_ids: torch.LongTensor
        of shape [rays_n]. Mesh triangle ids, MeshRenderer.INVALID_ID if didn't hit any face.
    uvs: torch.Tensor
        of shape [rays_n, 2]. Barycentric coordinates of hit points, 0 if didn't hit any face.
    xyz_world, xyz_cam : torch.Tensor
        of shape [rays_n, 3]. Coordinates of hit points, inf if didn't hit any face.
    """
    def __init__(self, renderer, depth_ray, normals_world, tri_ids, uvs):
        self.depth_ray = depth_ray
        self.normals_world = normals_world
        self.ray_dirs_world = renderer.w_rays[:, 3:6]
        self.ray_dirs_cam = renderer.c_rays
        self.tri_ids = tri_ids
        self.uvs = uvs

        self._c2w_rot = renderer.c2w_rot
        self._w_ray_origins = renderer.w_rays[:, :3]

    @cached_property
    def depth_z(self):
        r"""Calculates cam-space z-depth.

        Depends on
        ----------
        depth_ray : torch.Tensor
            of shape [rays_n].
        ray_dirs_cam: torch.Tensor
            of shape [rays_n, 3].

        Returns
        -------
        depth_z : torch.Tensor
            of shape [rays_n], inf if didn't hit any face.
        """
        return self.depth_ray * self.ray_dirs_cam[:, 2]

    @cached_property
    def is_frontfacing(self):
        r"""Tests if the faces are oriented towards the camera.

        Depends on
        ----------
        normals_world : torch.Tensor
            of shape [rays_n, 3].
        ray_dirs_world : torch.Tensor
            of shape [rays_n, 3].

        Returns
        -------
        is_frontfacing : torch.BoolTensor
            of shape [rays_n], False if didn't hit any face.
        """
        cos = (self.normals_world.unsqueeze(1) @ self.ray_dirs_world.unsqueeze(2)).squeeze(2).squeeze(1)
        return cos < 0

    @cached_property
    def normals_cam(self):
        r"""Calculates cam-space normals.

        Depends on
        ----------
        normals_world: torch.LongTensor
            of shape [rays_n, 3].

        Returns
        -------
        normals_cam : torch.Tensor
            of shape [rays_n, 3], 0 if didn't hit any face.
        """
        return self.normals_world @ self._c2w_rot

    @cached_property
    def xyz_cam(self):
        r"""Calculates cam-space coordinates of hit points.

        Depends on
        ----------
        depth_ray : torch.Tensor
            of shape [rays_n].
        ray_dirs_cam : torch.Tensor
            of shape [rays_n, 3].

        Returns
        -------
        xyz_cam : torch.Tensor
            of shape [rays_n, 3], inf if didn't hit any face.
        """
        return self.ray_dirs_cam * self.depth_ray.unsqueeze(-1)

    @cached_property
    def xyz_world(self):
        r"""Calculates world-space coordinates of hit points.

        Depends on
        ----------
        depth_ray : torch.Tensor
            of shape [rays_n].
        ray_dirs_world : torch.Tensor
            of shape [rays_n, 3].

        Returns
        -------
        xyz_world : torch.Tensor
            of shape [rays, 3], inf if didn't hit any face.
        """
        return self.ray_dirs_world.mul(self.depth_ray.unsqueeze(1)).add_(self._w_ray_origins)


class Render(CachedProps):
    r"""Represents a rendering.

    Attributes
    ---------
    attr_samples : list of torch.Tensor
        each of shape [height, width, attrs_n]. Interpolated vertex attributes, random if didn't hit any face, NaN if invalid.
    depth_ray, depth_z: torch.Tensor
        of shape [height, width]. Depth map, inf if didn't hit any face, NaN if invalid.
    is_frontfacing: torch.BoolTensor
        of shape [height, width]. True for faces oriented towards the camera, False if didn't hit any face.
    normals_world, normals_cam: torch.Tensor
        of shape [height, width, 3]. Normal map, 0 if didn't hit any face, NaN if invalid.
    ray_dirs_world, ray_dirs_cam: torch.Tensor
        of shape [height, width, 3]. Directions of rays passing through centers of pixels, NaN if invalid.
    tri_ids: torch.LongTensor
        of shape [height, width]. Mesh triangle ids, MeshRenderer.INVALID_ID if didn't hit any face or invalid.
    uvs: torch.Tensor
        of shape [height, width, 2]. Barycentric coordinates of hit points, 0 if didn't hit any face, NaN if invalid.
    xyz_world, xyz_cam : torch.Tensor
        of shape [height, width, 3]. Coordinates of hit points, inf if didn't hit any face, NaN if invalid.
    """
    def __init__(self, renderer, ray_render):
        self.renderer = renderer
        self.ray_render = ray_render
        self.attr_samples = None

    def __getattr__(self, attr):
        if attr in {'depth_ray', 'depth_z', 'normals_cam', 'normals_world',
                    'ray_dirs_cam', 'ray_dirs_world', 'uvs', 'xyz_cam', 'xyz_world'}:
            self.__dict__[attr] = self.renderer.rays_to_pixels(getattr(self.ray_render, attr))
        elif attr == 'is_frontfacing':
            self.__dict__['is_frontfacing'] = self.renderer.rays_to_pixels(self.ray_render.is_frontfacing, False)
        elif attr == 'tri_ids':
            self.__dict__['tri_ids'] = self.renderer.rays_to_pixels(self.ray_render.tri_ids, MeshRenderer.INVALID_ID)
        return self.__getattribute__(attr)
