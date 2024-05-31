from functools import cached_property

import numpy as np
import open3d as o3d
import torch

from ovl.utils import CachedProps


class MeshRenderer(CachedProps):
    r"""Triangle mesh renderer based on Intel Embree ray-tracer.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The mesh to render.

    Attributes
    ----------
    mesh : o3d.t.geometry.TriangleMesh
        The mesh being rendered.
    tris : torch.IntTensor
        of shape [tris_n, 3]. Vertex index triplets that form the triangles.
    """
    device = 'cpu'
    dtype = torch.float32
    INVALID_ID = o3d.t.geometry.RaycastingScene.INVALID_ID

    def __init__(self, mesh):
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.raycasting = o3d.t.geometry.RaycastingScene()
        self.raycasting.add_triangles(self.mesh)

    @cached_property
    def tris(self):
        return torch.from_numpy(self.mesh.triangle['indices'].numpy())

    def render_rays(self, rays, cull_backface=False):
        r"""Renders the mesh along the rays.

        Parameters
        ----------
        rays : torVch.Tensor
            of shape [rays_n, 6]. Each one is (origin_x, _y, _z, direction_x, _y, _z).
        cull_backface : bool
            If True, render the back of the faces as empty.

        Returns
        -------
        render : Render
        """
        rays_t = o3d.core.Tensor.from_numpy(rays.numpy())
        result = self.raycasting.cast_rays(rays_t); del rays_t
        render = Render(
            torch.from_numpy(result['t_hit'].numpy()),
            torch.from_numpy(result['primitive_normals'].numpy()),
            torch.from_numpy(result['primitive_ids'].numpy().astype(np.int64)),
            torch.from_numpy(result['primitive_uvs'].numpy()),
        ); del result

        if cull_backface:
            render.ray_dirs = rays[:, 3:6]
            is_frontfacing = render.is_frontfacing
            render = Render(
                render.hit_depth.where(is_frontfacing, render.hit_depth.new_tensor(float('inf'))),
                render.normals.where(is_frontfacing.unsqueeze(1), render.normals.new_tensor(0)),
                render.tri_ids.where(is_frontfacing, render.tri_ids.new_tensor(self.INVALID_ID)),
                render.uvs.where(is_frontfacing.unsqueeze(1), render.uvs.new_tensor(0)),
            ); del is_frontfacing
        del rays
        return render

    @staticmethod
    def get_hit_xyz(rays, hit_depth):
        r"""Calculates coordinates of hit points.

        Parameters
        ----------
        rays : torch.Tensor
            of shape [rays_n, 6]. Each one is (origin_x, _y, _z, direction_x, _y, _z).
        hit_depth : torch.Tensor
            of shape [rays_n]. Distance to the first hit.

        Returns
        -------
        xyz : torch.Tensor
            of shape [rays_n, 3].
        """
        xyz = rays[:, 3:6].mul(hit_depth.unsqueeze(1)).add_(rays[:, :3])
        return xyz

    def test_ray_intersection(self, rays, ray_lens, occ_thres=0):
        r"""Checks if ray segments intersect the mesh.

        Parameters
        ----------
        rays : torch.Tensor
            of shape [rays_n, 6]. Each one is (origin_x, _y, _z, direction_x, _y, _z).
        ray_lens : torch.Tensor
            of shape [rays_n]. Lengths of the rays in ray_dir units.
        occ_thres : float
            The intersection counts only if the ray ends deeper than this below the surface.

        Returns
        -------
        intersects : torch.BoolTensor
            of shape [rays_n].

        Notes
        -----
        Intersections with both front- and back-faces count.
        """
        hit_depths = self.render_rays(rays).hit_depth
        hit_depths_shifted = hit_depths.to(ray_lens).add_(occ_thres); del hit_depths
        intersects = ray_lens <= hit_depths_shifted; del ray_lens, hit_depths_shifted
        return intersects

    def test_pts_visibility(self, pts, viewpoint, occ_thres=0, div_eps=1e-12):
        r"""Checks if points are visible from a viewpoint and not occluded by the mesh.

        Parameters
        ----------
        pts : torch.Tensor
            of shape [pts_n, 3].
        viewpoint : torch.Tensor
            of shape [3].
        occ_thres : float
            Points deeper than this below the surface are occluded.

        Returns
        -------
        is_visible : torch.BoolTensor
            of shape [pts_n].
        """
        rays = pts.new_empty([len(pts), 6])
        origins = rays[:, :3]
        origins.copy_(viewpoint); del origins

        dirs = rays[:, 3:6]
        torch.sub(pts, viewpoint, out=dirs); del viewpoint
        pt_depths = dirs.norm(dim=1)
        dirs /= pt_depths.clamp(min=div_eps).unsqueeze(1); del dirs
        return self.test_ray_intersection(rays, pt_depths, occ_thres)

    def interpolate(self, attrs, tri_ids, uvs):
        r"""Interpolates mesh vertex attributes at hit points.

        Parameters
        ----------
        attrs : iterable of torch.Tensor
            each of shape [verts_n, attrs_n].
        tri_ids : torch.Tensor
            of shape [rays_n]. Mesh triangle ids.
        uvs : torch.Tensor
            of shape [rays_n, 2]. Barycentric coordinates of hit points.

        Returns
        -------
        samples : list of torch.Tensor
            each of shape [rays_n, attrs_n]. If tri_id is invalid, the value is random.
        """
        rays_n = tri_ids.shape[0]
        bar_weights = uvs.new_empty(rays_n, 3)
        torch.sub(1, uvs.sum(1), out=bar_weights[:, 0])
        bar_weights[:, 1:3] = uvs; del uvs

        tri_ids = tri_ids.where(tri_ids != MeshRenderer.INVALID_ID, tri_ids.new_tensor(0))
        vert_ids = self.tris[tri_ids]; del tri_ids

        samples = []
        for attrs_sub in attrs:
            tri_attrs = attrs_sub[vert_ids.view(-1)].view(rays_n, 3, -1); del attrs_sub
            samples_sub = (bar_weights.unsqueeze(1) @ tri_attrs).squeeze(1); del tri_attrs
            samples.append(samples_sub); del samples_sub
        return samples


class Render(CachedProps):
    r"""Represents ray-traced values.

    Attributes
    ----------
    hit_depth : torch.Tensor
        of shape [rays_n]. Distance to the first hit, in ray_dir units, inf if didn't hit any face.
    is_frontfacing: torch.BoolTensor
        of shape [rays_n]. True for faces oriented towards the camera, False if didn't hit any face.
    normals : torch.Tensor
        of shape [rays_n, 3]. Normals at hit points, 0 if didn't hit any face.
    ray_dirs : torch.Tensor
        of shape [rays_n, 3]. Directions of the traced rays.
    tri_ids : torch.Tensor
        of shape [rays_n]. Mesh triangle ids, MeshRenderer.INVALID_ID if didn't hit any face.
    uvs : torch.Tensor
        of shape [rays_n, 2]. Barycentric coordinates of hit points, 0 if didn't hit any face.
    """
    def __init__(self, hit_depth, normals, tri_ids, uvs):
        self.hit_depth = hit_depth
        self.normals = normals
        self.tri_ids = tri_ids
        self.uvs = uvs
        self.ray_dirs = None

    @cached_property
    def is_frontfacing(self):
        r"""Tests if the faces are oriented towards the ray origins.

        Depends on
        ----------
        normals : torch.Tensor
            of shape [rays_n, 3].
        ray_dirs : torch.Tensor
            of shape [rays_n, 3].

        Returns
        -------
        is_frontfacing : torch.BoolTensor
            of shape [rays_n], 0 if didn't hit any face.
        """
        cos = (self.normals.unsqueeze(1) @ self.ray_dirs.unsqueeze(2)).squeeze(2).squeeze(1)
        return cos < 0
