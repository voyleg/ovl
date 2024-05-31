from functools import cached_property

import nvdiffrast.torch as dr
import numpy as np
import open3d as o3d
import torch

from ovl.utils import CachedProps
from ovl.camera.camera_model import PinholeCameraModel


class MeshRenderer(CachedProps):
    r"""Triangle mesh rasterizer based on nvdiffrast.

    Parameters
    ----------
    device : torch.device

    Attributes
    ----------
    c_rays : torch.Tensor
        of shape [height, width, 3]. Cam-space directions of optical rays passing through centers of the pixels.
    c2w_t : torch.Tensor
        of shape [3]. Camera-to-world translation vector.
    cam_model : PinholeCameraModel
        Camera model.
    device : torch.device
    far : float
        Distance to the far plane.
    near : float
        Distance to the near plane.
    mesh : o3d.geometry.TriangleMesh
        The mesh being rendered.
    normals : torch.Tensor
        of shape [tris_n, 3]. World-space triangle normals.
    size_hw : tuple of int
        (h, w). Image dimensions.
    ssaa : int
        Square root of the number of supersamples in supersampling anti-aliasing.
    tris : torch.IntTensor
        of shape [tris_n, 3]. Vertex index triplets that form the triangles.
    w2c_rot : torch.Tensor
        of shape [3, 3]. World-to-camera rotation matrix.
    w2clip : torch.Tensor
        of shape [4, 4]. World-to-clip space transform matrix.
    xyz : torch.Tensor
        of shape [verts_n, 3]. World-space vertex coords.
    xyzw_clip : torch.Tensor
        of shape [verts_n, 4]. Clip-space vertex coords.
    """
    dtype = torch.float32

    def __init__(self, device='cuda'):
        self.device = device
        self.glctx = dr.RasterizeGLContext(output_db=False, device=device)

        self.mesh, self.normals, self.tris, self.xyz = None, None, None, None  # set_mesh
        self.cam_model, self.far, self.near, self.proj_mat = None, None, None, None  # set_cam_model
        self.c_rays, self.size_hw, self.ssaa = None, None, None  # set_resolution
        self.c2w_t, self.w2c_rot, self.w2clip = None, None, None  # set_cam_pose

    def set_mesh(self, mesh, set_normals=True):
        r"""Sets the rasterized mesh.

        Parameters
        ----------
        mesh : o3d.geometry.TriangleMesh
            The mesh to render. It should be relatively close to the origin,
            otherwise the rasterization is buggy, presumably, due to floating-point errors.
        set_normals : bool
            Set to True, if you plan to render normals or use backface culling.
        """
        self.free_cache('xyzw_clip')
        self.mesh = o3d.geometry.TriangleMesh(mesh)
        self.xyz = torch.from_numpy(np.asarray(self.mesh.vertices)).to(self.device, self.dtype)
        self.tris = torch.from_numpy(np.asarray(self.mesh.triangles)).to(self.device, torch.int)
        if set_normals:
            self.mesh.compute_triangle_normals()
            self.normals = torch.from_numpy(np.asarray(self.mesh.triangle_normals)).to(self.device, self.dtype)
        else:
            self.normals = None

    def set_cam_model(self, cam_model, near=.5, far=1.5):
        r"""Sets up OpengGL projection matrix, as described at http://www.songho.ca/opengl/gl_projectionmatrix.html
            Call set_resolution afterwards.

        Parameters
        ----------
        cam_model : PinholeCameraModel
            Camera model.
        near : float
            Distance to the near plane.
        far : float
            Distance to the far plane.
        """
        self.c_rays, self.size_hw, self.ssaa = None, None, None
        self.near = near
        self.far = far

        self.cam_model = cam_model.clone().to(self.device, self.dtype)
        lb = -cam_model.principal / cam_model.focal * near
        rt = (cam_model.size_wh - cam_model.principal) / cam_model.focal * near

        proj_mat = torch.zeros(4, 4, device=self.device, dtype=self.dtype)
        proj_mat[[0, 1], [0, 1]] = (2 * near / (rt - lb)).to(self.device, self.dtype)
        proj_mat[:2, 2] = ((lb + rt) / (rt - lb)).to(self.device, self.dtype)
        proj_mat[2, 2] = - (far + near) / (far - near)
        proj_mat[2, 3] = - 2 * far * near / (far - near)
        proj_mat[3, 2] = -1
        # Z-axis in camera space in OpenGL points to camera, and in PinholeCameraModel it points from camera,
        # so we flip it.
        proj_mat[:, 2] *= -1
        # Y-axis should also be flipped, since in OpenGL it points up, and in PinholeCameraModel it points down.
        # However, the vertical orientation of the rendered image also differs, so we keep the orientation of the Y-axis
        # to negate the flipping of the image.
        self.proj_mat = proj_mat

    def set_resolution(self, h, w, ssaa=None):
        r"""Sets the resolution of render.

        Parameters
        ----------
        h : int
        w : int
        ssaa : int
            If set, render with supersampling anti-aliasing with uniform grid of ssaa^2 supersamples.
        """
        self.ssaa = ssaa
        if ssaa is None:
            self.size_hw = (h, w)
        else:
            self.size_hw = (h * ssaa, w * ssaa)
        new_wh = self.cam_model.focal.new_tensor([w, h], dtype=torch.int)
        self.cam_model.resize_(new_wh); del new_wh

        rays = self.cam_model.get_pix_rays()
        self.c_rays = rays.permute(1, 2, 0).to(self.device, self.dtype).contiguous(); del rays

    def set_cam_pose(self, w2c):
        r"""Sets world-to-cam transform.

        Parameters
        ----------
        w2c : torch.Tensor
            of shape [4, 4]. World-to-camera transform matrix.
        """
        self.free_cache('xyzw_clip')
        self.w2clip = self.proj_mat @ w2c
        self.w2c_rot = w2c[:3, :3]
        self.c2w_t = w2c.inverse()[:3, 3]

    @cached_property
    def xyzw_clip(self):
        return (self.xyz @ self.w2clip.T[:3]).add_(self.w2clip[:, 3])

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
        rast = self.rasterize()
        if cull_backface:
            render = Render(self, rast)
            is_frontfacing = render.is_frontfacing; del render
            rast = rast.where(is_frontfacing.unsqueeze(2), rast.new_tensor(0)); del is_frontfacing
        render = Render(self, rast)
        if attrs is not None:
            render.attr_samples = self.interpolate(attrs, rast)
        del rast
        return render

    def rasterize(self):
        r"""Rasterizes the mesh.

        Returns
        -------
        rast : torch.Tensor
            of shape [height, width, 4]. Each pixel is (u, v, z/w, triangle_id), all 0 if didn't hit any face.
        """
        rast, _ = dr.rasterize(self.glctx, self.xyzw_clip.unsqueeze(0), self.tris, self.size_hw); del _
        rast = rast.squeeze(0)
        return rast

    def interpolate(self, attrs, rast):
        r"""Interpolates mesh vertex attributes at hit points.

        Parameters
        ----------
        attrs : iterable of torch.Tensor
            each of shape [verts_n, attrs_n].
        rast : torch.Tensor
            of shape [height, width, 4]. The output of rasterize.

        Returns
        -------
        samples : list of torch.Tensor
            each of shape [height, width, attrs_n], random if didn't hit any face.
        """
        samples = []
        for attrs_sub in attrs:
            samples_sub, _ = dr.interpolate(attrs_sub.unsqueeze(0), rast.unsqueeze(0), self.tris); del _
            samples_sub = samples_sub.squeeze(0)
            samples.append(samples_sub); del samples_sub
        return samples

    def antialias(self, img):
        r"""Downsamples the image rendered with supersampling.

        Parameters
        ----------
        img : torch.Tensor
            of shape [channels_n, height * ssaa, width * ssaa].

        Returns
        -------
        img_d : torch.Tensor
            of shape [channels_n, height, width].
        """
        img_d = torch.nn.functional.avg_pool2d(img.unsqueeze(0), self.ssaa).squeeze(0)
        return img_d


class Render(CachedProps):
    r"""Represents a rendering.

    Parameters
    ----------
    renderer : MeshRenderer
    rast : torch.Tensor
        of shape [height, width, 4]. The output of MeshRenderer.rasterize.

    Attributes
    ----------
    attr_samples : list of torch.Tensor
        each of shape [height, width, attrs_n]. Interpolated vertex attributes, random if didn't hit any face.
    depth_ray, depth_z: torch.Tensor
        of shape [height, width]. Depth map, random if didn't hit any face.
    is_frontfacing: torch.BoolTensor
        of shape [height, width]. True for faces oriented towards the camera, random if didn't hit any face.
    normals_world, normals_cam: torch.Tensor
        of shape [height, width, 3]. Normal map, random if didn't hit any face.
    ray_dirs_world, ray_dirs_cam: torch.Tensor
        of shape [height, width, 3]. Directions of rays passing through centers of pixels.
    tri_ids: torch.LongTensor
        of shape [height, width]. Mesh triangle ids, -1 if didn't hit any face.
    uvs: torch.Tensor
        of shape [height, width, 2]. Barycentric coordinates of hit points, 0 if didn't hit any face.
    xyz_world, xyz_cam : torch.Tensor
        of shape [height, width, 3]. Coordinates of hit points, random value if didn't hit any face.

    ndc_z : torch.Tensor
        of shape [height, width]. Normalized depth z/w, 0 if didn't hit any face.
    tri_ids_offs: torch.LongTensor
        of shape [height, width]. Mesh triangle ids minus one, 0 if didn't hit any face.
    """
    def __init__(self, renderer, rast):
        self.uvs = rast[..., :2]
        self.ndc_z = rast[..., 2]
        self.tri_ids_offs = rast[..., 3]
        self.ray_dirs_cam = renderer.c_rays

        self._near = renderer.near
        self._far = renderer.far
        self._normals = renderer.normals
        self._w2c_rot = renderer.w2c_rot
        self._c2w_t = renderer.c2w_t

        self.attr_samples = None

    @cached_property
    def depth_z(self):
        r"""Calculates cam-space z-depth.

        Depends on
        ----------
        ndc_z : torch.Tensor
            of shape [height, width].

        Returns
        -------
        depth_z : torch.Tensor
            of shape [height, width]. If didn't hit any face, the value is (2 * far * near) / (far + near).
        """
        near, far = self._near, self._far
        depth_z = self.ndc_z.mul(near - far).add_(far + near)
        depth_z = depth_z.reciprocal_().mul_(2 * far * near)
        return depth_z

    @cached_property
    def depth_ray(self):
        r"""Calculates distance to the first hit along the ray.

        Depends on
        ----------
        depth_z : torch.Tensor
            of shape [height, width]. Cam-space z-depth.
        ray_dirs_cam : torch.Tensor
            of shape [height, width, 3].

        Returns
        -------
        depth_ray : torch.Tensor
            of shape [height, width], random if didn't hit any face.
        """
        return self.depth_z / self.ray_dirs_cam[..., 2]

    @cached_property
    def is_frontfacing(self):
        r"""Tests if the faces are oriented towards the camera.

        Depends on
        ----------
        normals_cam : torch.Tensor
            of shape [height, width, 3].
        ray_dirs_cam : torch.Tensor
            of shape [height, width, 3].

        Returns
        -------
        is_frontfacing : torch.BoolTensor
            of shape [height, width], random if didn't hit any face.
        """
        cos = (self.normals_cam.unsqueeze(2) @ self.ray_dirs_cam.unsqueeze(3)).squeeze(3).squeeze(2)
        return cos < 0

    @cached_property
    def normals_world(self):
        r"""Calculates world-space normal map.

        Depends on
        ----------
        tri_ids: torch.LongTensor
            of shape [height, width].

        Returns
        -------
        normals_world : torch.Tensor
            of shape [height, width, 3], random value if didn't hit any face.
        """
        return self._normals[self.tri_ids.view(-1)].reshape(*self.tri_ids.shape, 3)

    @cached_property
    def normals_cam(self):
        r"""Calculates cam-space normal map.

        Depends on
        ----------
        normals_world: torch.LongTensor
            of shape [height, width, 3].

        Returns
        -------
        normals_cam : torch.Tensor
            of shape [height, width, 3], random value if didn't hit any face.
        """
        return self.normals_world @ self._w2c_rot.T

    @cached_property
    def ray_dirs_world(self):
        r"""Calculates world-space optical rays passing through centers of pixels.

        Depends on
        ----------
        ray_dirs_cam : torch.Tensor
            of shape [height, width, 3].

        Returns
        -------
        ray_dirs_world : torch.Tensor
            of shape [height, width, 3].
        """
        return self.ray_dirs_cam @ self._w2c_rot

    @cached_property
    def tri_ids(self):
        return self.tri_ids_offs.long().sub_(1)

    @cached_property
    def xyz_cam(self):
        r"""Calculates cam-space coordinates of hit points.

        Depends on
        ----------
        depth_ray : torch.Tensor
            of shape [height, width]
        ray_dirs_cam : torch.Tensor
            of shape [height, width, 3]

        Returns
        -------
        xyz_cam : torch.Tensor
            of shape [height, width, 3], random if didn't hit any face.
        """
        return self.ray_dirs_cam * self.depth_ray.unsqueeze(-1)

    @cached_property
    def xyz_world(self):
        r"""Calculates world-space coordinates of hit points.

        Depends on
        ----------
        xyz_cam : torch.Tensor
            of shape [height, width, 3].

        Returns
        -------
        xyz_world : torch.Tensor
            of shape [height, width, 3], random if didn't hit any face.
        """
        return (self.xyz_cam @ self._w2c_rot).add_(self._c2w_t)
