"""Write per-patch QuadWild edge-size multipliers from a painted vertex group.

The patched quad_from_patches engine (0003) reads `<traced mesh>.scales`,
with one multiplier for each patch ID in `<traced mesh>.patch`.
"""

import os

from mathutils.kdtree import KDTree


DENSITY_GROUP = "qremeshify_density"


def patch_scale(weights):
    """Harmonic mean of (0.01 + weight), with unassigned weights = zero.

    A low weight requires a short edge (dense mesh); the reciprocal mean
    deliberately makes a small number of fine-detail vertices influential.
    """
    if not weights:
        return 1.0
    return len(weights) / sum(1.0 / (0.01 + max(0.0, min(1.0, w))) for w in weights)


def _read_traced_patches(traced_path):
    """Read triangulated traced OBJ and its parallel zero-based patch IDs."""
    vertices, faces = [], []
    with open(traced_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append(tuple(float(value) for value in line.split()[1:4]))
            elif line.startswith("f "):
                # OBJ face tokens can be v, v/vt or v/vt/vn.
                faces.append(tuple(int(tok.split("/")[0]) - 1 for tok in line.split()[1:]))

    patch_path = os.path.splitext(traced_path)[0] + ".patch"
    with open(patch_path, encoding="utf-8") as handle:
        declared = int(handle.readline().strip())
        patch_ids = [int(token) for line in handle for token in line.split()]
    if not faces or len(patch_ids) != len(faces) or declared != len(faces):
        raise ValueError("Traced mesh and patch file have different face counts")
    if min(patch_ids) < 0 or max(patch_ids) >= len(faces):
        raise ValueError("Patch identifiers are outside the expected zero-based range")
    if any(any(index < 0 or index >= len(vertices) for index in face) for face in faces):
        raise ValueError("Traced mesh contains invalid vertex indices")
    return vertices, faces, patch_ids


def write_patch_scales(traced_path, source_obj, depsgraph, transform, symmetry_axes=(False, False, False)):
    """Sample the painted source group at each *unique* traced patch vertex.

    Use Blender's native KDTree to match traced vertices to evaluated source
    vertices in the same rotated/scaled coordinates as the exported OBJ.
    When the group is absent, remove stale scale files and use global density.
    """
    scales_path = os.path.splitext(traced_path)[0] + ".scales"
    group = source_obj.vertex_groups.get(DENSITY_GROUP)
    if group is None:
        if os.path.isfile(scales_path):
            os.remove(scales_path)
        return 0

    vertices, faces, patch_ids = _read_traced_patches(traced_path)
    evaluated = source_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        if not evaluated_mesh.vertices:
            raise ValueError("The source has no vertices to sample density from")
        index = KDTree(len(evaluated_mesh.vertices))
        weights = {}
        for vertex in evaluated_mesh.vertices:
            # Match the same positive half used by QRemeshify's symmetry
            # bisect. Otherwise mirrored counterparts can contaminate paint.
            if any(symmetry_axes[axis] and vertex.co[axis] < -1e-6
                   for axis in range(3)):
                continue
            index.insert(transform @ vertex.co, vertex.index)
            weights[vertex.index] = next((assignment.weight for assignment in vertex.groups
                                          if assignment.group == group.index), 0.0)
        if not weights:
            raise ValueError("No source vertices remain on the selected symmetry side")
        index.balance()

        # A shared patch vertex should contribute only once, irrespective of
        # its triangle valence. Missing groups/weights are treated as weight 0.
        patch_vertices = [set() for _ in range(max(patch_ids) + 1)]
        for face, patch_id in zip(faces, patch_ids):
            patch_vertices[patch_id].update(face)
        sampled = [weights[index.find(vertex)[1]] for vertex in vertices]
        multipliers = [patch_scale([sampled[i] for i in indices])
                       for indices in patch_vertices]
    finally:
        evaluated.to_mesh_clear()

    # Atomic replacement avoids handing an incomplete scale file to a native
    # solver. An empty patch ID defaults to the global edge-size multiplier.
    temporary_path = scales_path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            for multiplier in multipliers:
                handle.write(f"{multiplier:.9f}\n")
        os.replace(temporary_path, scales_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return len(multipliers)
