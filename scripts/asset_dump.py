import jax
from jax import random, numpy as jp
import numpy as np
import pygltflib
import pywavefront

from madrona_mjx import BatchRenderer
import argparse
from time import time

import sys

def load_obj(filename):
    vertices = []
    uvs = []
    normals = []
    indices = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.split()[1:])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                for triple in line.split()[1:]:
                    ints = list(map(lambda x: int(x) - 1, triple.split('/')))
                    indices.append(ints[0])

    return vertices, uvs, normals, indices


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dump-path', type=str, required=True)

args = arg_parser.parse_args()

objs = []
objs.append(load_obj('data/sphere.obj'))
objs.append(load_obj('data/plane.obj'))
objs.append(load_obj('data/cube_render.obj'))
objs.append(load_obj('data/wall_render.obj'))
objs.append(load_obj('data/agent_render.obj'))
objs.append(load_obj('data/ramp_render.obj'))
objs.append(load_obj('data/elongated_render.obj'))

mesh_verts = []
mesh_idxs = []

mesh_vert_offsets = []
mesh_idx_offsets = []

for o in objs:
    verts, uvs, normals, indices = o

    mesh_vert_offsets.append(len(mesh_verts))
    mesh_idx_offsets.append(len(mesh_idxs))

    mesh_verts += verts
    mesh_idxs += indices
    
mesh_verts = np.array(mesh_verts, dtype=np.float32)
mesh_idxs = np.array(mesh_idxs, dtype=np.int32)

print(mesh_verts.shape)
print(mesh_idxs.shape)

mesh_vert_offsets = np.array(mesh_vert_offsets, dtype=np.int32)
mesh_idx_offsets = np.array(mesh_idx_offsets, dtype=np.int32)

gltf_buffer_views = []
gltf_accessors = []
gltf_meshes = []
gltf_instances = []

for i in range(len(objs)):
    vert_offset = mesh_vert_offsets[i]
    idx_offset = mesh_idx_offsets[i]
    if i == len(mesh_vert_offsets) - 1:
        num_verts = mesh_verts.shape[0] - vert_offset
        num_indices = mesh_idxs.shape[0] - idx_offset
    else:
        num_verts = mesh_vert_offsets[i + 1] - vert_offset
        num_indices = mesh_idx_offsets[i + 1] - idx_offset

    vert_offset = int(vert_offset)
    idx_offset = int(idx_offset)
    num_verts = int(num_verts)
    num_indices = int(num_indices)

    num_bytes_per_idx = 4
    num_bytes_per_vert = 12

    gltf_buffer_views.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=num_bytes_per_idx * idx_offset,
        byteLength=num_bytes_per_idx * num_indices,
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))

    gltf_accessors.append(pygltflib.Accessor(
        bufferView=len(gltf_buffer_views) - 1,
        componentType=pygltflib.UNSIGNED_INT,
        count=num_indices,
        type=pygltflib.SCALAR,
        max=[int(mesh_idxs.max())],
        min=[int(mesh_idxs.min())],
    ))

    gltf_buffer_views.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=num_bytes_per_idx * mesh_idxs.shape[0] + num_bytes_per_vert * vert_offset,
        byteLength=num_bytes_per_vert * num_verts,
        target=pygltflib.ARRAY_BUFFER,
    ))

    gltf_accessors.append(pygltflib.Accessor(
        bufferView=len(gltf_buffer_views) - 1,
        componentType=pygltflib.FLOAT,
        count=num_verts,
        type=pygltflib.VEC3,
        max=mesh_verts.max(axis=0).tolist(),
        min=mesh_verts.min(axis=0).tolist(),
    ))

    gltf_meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=len(gltf_accessors) - 1),
        indices=len(gltf_accessors) - 2,
        material=0,
    )]))

#for i in range(len(geom_sizes)):
#    geom_types = m.geom_type
#    geom_data_ids = m.geom_dataid
#    geom_sizes = jax.device_get(m.geom_size)
#
#    geom_type = geom_types[i]
#    geom_size = geom_sizes[i]
#
#    pos = geom_xpos[i]
#    mat = geom_xmat[i]
#
#    if geom_type == 7:
#        gltf_instances.append(pygltflib.Node(
#            mesh=int(geom_data_ids[i]),
#            matrix=[
#                float(mat[0][0]), float(mat[1][0]), float(mat[2][0]), 0,
#                float(mat[0][1]), float(mat[1][1]), float(mat[2][1]), 0,
#                float(mat[0][2]), float(mat[1][2]), float(mat[2][2]), 0,
#                float(pos[0]),    float(pos[1]),    float(pos[2]),    1,
#            ],
#        ))

# write to gltf

triangles_binary_blob = mesh_idxs.flatten().tobytes()
points_binary_blob = mesh_verts.tobytes()

gltf = pygltflib.GLTF2(
    scene=0,
    scenes=[pygltflib.Scene(nodes=list(range(len(gltf_instances))))],
    nodes=[],
    meshes=gltf_meshes,
    accessors=gltf_accessors,
    bufferViews=gltf_buffer_views,
    buffers=[
        pygltflib.Buffer(
            byteLength=len(triangles_binary_blob) + len(points_binary_blob)
        )
    ],
    materials=[
        pygltflib.Material(pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )),
    ],
)
gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)

gltf.save(args.dump_path)
