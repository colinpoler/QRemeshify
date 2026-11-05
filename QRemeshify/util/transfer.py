import bmesh
import bpy
import mathutils
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


def transfer_materials(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
    """Copy material slots and re-assign per-face material indices by nearest face.

    The remeshed geometry lives in the source's local space with rotation and
    scale baked in (see operator export), so the source is transformed by the
    same matrix before the nearest-face lookup.
    """
    if len(src_obj.material_slots) == 0:
        return

    dst_mesh = dst_obj.data
    for slot in src_obj.material_slots:
        dst_mesh.materials.append(slot.material)

    if len(src_obj.material_slots) < 2:
        return  # single slot: all polygons already point at index 0

    bm = bmesh.new()
    bm.from_mesh(src_obj.data)
    bmesh.ops.transform(bm, matrix=_rot_scale_matrix(src_obj), verts=bm.verts)
    # Triangulation makes BVH hit indices correspond to material indices in
    # material_by_face; splitting source faces does not change their materials.
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.faces.ensure_lookup_table()
    material_by_face = [f.material_index for f in bm.faces]
    tree = BVHTree.FromBMesh(bm)

    for poly in dst_mesh.polygons:
        hit = tree.find_nearest(poly.center)
        if hit is not None and hit[2] is not None:
            poly.material_index = material_by_face[hit[2]]

    bm.free()


def freestyle_edge_marks(mesh):
    """Read Freestyle edge marks on both legacy and attribute-based Blender APIs."""
    attribute = mesh.attributes.get("freestyle_edge")
    if attribute is not None:
        return [item.value for item in attribute.data]
    return [getattr(edge, "use_freestyle_mark", False) for edge in mesh.edges]


def _set_freestyle_marks(mesh, marks):
    """Write marks through the supported edge API for this Blender version."""
    attr = mesh.attributes.get("freestyle_edge")
    if attr is None and mesh.edges and not hasattr(mesh.edges[0], "use_freestyle_mark"):
        attr = mesh.attributes.new("freestyle_edge", 'BOOLEAN', 'EDGE')
    for edge, marked in zip(mesh.edges, marks):
        if attr is not None:
            attr.data[edge.index].value = bool(marked)
        else:
            edge.use_freestyle_mark = bool(marked)


def transfer_edge_overlays(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
    """Copy seam, sharp and Freestyle flags from aligned nearest-midpoint edges.

    KD-tree midpoint queries run in native code. An absolute cosine threshold
    of 0.7 rejects crossed edges while accepting reversed edge directions.
    """
    source_marks = freestyle_edge_marks(src_obj.data)
    # Blender 5.x stores Freestyle as a boolean EDGE attribute; older versions
    # expose MeshEdge.use_freestyle_mark instead.
    target_attribute = dst_obj.data.attributes.get("freestyle_edge")
    if (target_attribute is None and dst_obj.data.edges
            and not hasattr(dst_obj.data.edges[0], "use_freestyle_mark")):
        target_attribute = dst_obj.data.attributes.new("freestyle_edge", 'BOOLEAN', 'EDGE')
    for destination, source in _matched_edges(src_obj, dst_obj):
        destination.use_seam = source.use_seam
        destination.use_edge_sharp = source.use_edge_sharp
        marked = bool(source_marks[source.index])
        if target_attribute is not None:
            target_attribute.data[destination.index].value = marked
        else:
            destination.use_freestyle_mark = marked
    dst_obj.data.update()


def transfer_uvs(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
    """UV transfer by independent face-corner projection, not Data Transfer.

    BVH indexes source triangles (which retain original per-loop UVs). Seam
    half-planes reject faces across an incident destination seam before nearest
    face selection. A small interior tie-break handles coincident seam vertices.
    """
    src_mesh, dst_mesh = src_obj.data, dst_obj.data
    if not src_mesh.uv_layers or not src_mesh.polygons or not dst_mesh.polygons:
        return

    # UVs are stored on face corners (loops), never per mesh vertex.
    # Create the destination layers here so this function is self-contained.
    for layer in src_mesh.uv_layers:
        if dst_mesh.uv_layers.get(layer.name) is None:
            dst_mesh.uv_layers.new(name=layer.name, do_init=False)

    src_mesh.calc_loop_triangles()
    triangles = src_mesh.loop_triangles
    if not triangles:
        return
    transform = _rot_scale_matrix(src_obj)
    src_coords = [transform @ vertex.co for vertex in src_mesh.vertices]
    tri_verts = [tuple(tri.vertices) for tri in triangles]
    tree = BVHTree.FromPolygons(src_coords, tri_verts, all_triangles=True)
    # The source BVH and destination corners share the exporter's baked
    # rotation/scale coordinate space. The source mesh is never modified.
    triangle_centers = [sum((src_coords[i] for i in ids), mathutils.Vector((0, 0, 0))) / 3.0 for ids in tri_verts]
    triangle_sizes = [max((src_coords[ids[i]] - src_coords[ids[(i + 1) % 3]]).length for i in range(3)) for ids in tri_verts]
    src_uvs = [
        (layer, src_mesh.uv_layers.get(layer.name)) for layer in dst_mesh.uv_layers if src_mesh.uv_layers.get(layer.name)
    ]
    if not src_uvs:
        return

    # Compute seam constraints from source edge matches, even if the user
    # disabled Keep Edge Overlays. UV-only mode never changes edge flags.
    seam_edges = {dst_edge.index for dst_edge, src_edge in _matched_edges(src_obj, dst_obj) if src_edge.use_seam}
    for poly in dst_mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex = dst_mesh.vertices[dst_mesh.loops[loop_index].vertex_index].co
            constraints = _uv_corner_outwards(dst_mesh, poly, loop_index, seam_edges)
            nearest = tree.find_nearest(vertex)
            if nearest is None or nearest[2] is None:
                continue

            # Gather nearby candidate source triangles. Unlike per-vertex
            # transfer, every loop is processed separately, preserving distinct
            # UV values for multiple islands at a shared mesh vertex.
            radius = nearest[3] + max(triangle_sizes[nearest[2]] * 1.5, 1e-6)
            candidates = tree.find_nearest_range(vertex, radius)
            hint = vertex + (poly.center - vertex) * 0.02
            selected = None
            for point, _normal, tri_index, distance in candidates:
                if tri_index is None:
                    continue
                center_offset = triangle_centers[tri_index] - vertex
                if any(center_offset.dot(outward) > 1e-7 for outward in constraints):
                    continue
                # The primary metric is vertex-to-face distance; the hint
                # resolves ties at a seam in favor of the same face side.
                ids = tri_verts[tri_index]
                hint_point = mathutils.geometry.closest_point_on_tri(hint, *(src_coords[i] for i in ids))
                score = distance + (hint_point - hint).length * 0.1
                if selected is None or score < selected[0]:
                    selected = (score, tri_index, point)

            if selected is None and constraints:
                # A large source/destination topology mismatch may require
                # searching farther for the matching side of the seam.
                for point, _normal, tri_index, distance in tree.find_nearest_range(vertex, radius * 4.0):
                    if tri_index is None or any(
                        (triangle_centers[tri_index] - vertex).dot(outward) > 1e-7 for outward in constraints
                    ):
                        continue
                    ids = tri_verts[tri_index]
                    hint_point = mathutils.geometry.closest_point_on_tri(hint, *(src_coords[i] for i in ids))
                    score = distance + (hint_point - hint).length * 0.1
                    if selected is None or score < selected[0]:
                        selected = (score, tri_index, point)
            if selected is None:
                # No candidate satisfies the seam-side constraints. Retain
                # nearest sampling as a fallback rather than leaving zero UVs.
                selected = (nearest[3], nearest[2], nearest[0])

            _, tri_index, point = selected
            ia, ib, ic = tri_verts[tri_index]
            weights = _uv_barycentric(point, src_coords[ia], src_coords[ib], src_coords[ic])
            source_loops = triangles[tri_index].loops
            for dst_layer, src_layer in src_uvs:
                uv0, uv1, uv2 = (src_layer.data[i].uv for i in source_loops)
                dst_layer.data[loop_index].uv = uv0 * weights[0] + uv1 * weights[1] + uv2 * weights[2]
    dst_mesh.update()


def transfer_vertex_groups(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> bool:
    """Transfer deform weights using Blender's native Data Transfer modifier.

    Only vertex group weights are enabled here: UVs and edge flags are handled
    by the explicit geometry-matching functions, not by Data Transfer. The
    modifier is applied immediately so temporary symmetry sources can be freed.
    Returns False when Blender cannot apply the modifier.
    """
    if not src_obj.vertex_groups:
        return True

    # Match layers by group name, rather than by potentially different indices.
    for group in src_obj.vertex_groups:
        if group.name not in dst_obj.vertex_groups:
            dst_obj.vertex_groups.new(name=group.name)

    mod = dst_obj.modifiers.new("QRemeshify Vertex Groups", "DATA_TRANSFER")
    mod.object = src_obj
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"

    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return True
    except RuntimeError:
        # Never retain a modifier targeting a soon-to-be-deleted bisected source.
        dst_obj.modifiers.remove(mod)
        return False


def copy_rig(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
    """Copy parenting and Armature modifiers so a rigged mesh stays rigged.

    Call only when vertex groups are retained; the binding then follows the
    independently transferred deform weights.
    """
    if src_obj.parent is not None:
        dst_obj.parent = src_obj.parent
        dst_obj.parent_type = src_obj.parent_type
        if src_obj.parent_type == "BONE":
            dst_obj.parent_bone = src_obj.parent_bone
        dst_obj.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()

    for mod in src_obj.modifiers:
        if mod.type == "ARMATURE":
            arm = dst_obj.modifiers.new(mod.name, "ARMATURE")
            arm.object = mod.object
            arm.use_vertex_groups = mod.use_vertex_groups
            arm.use_bone_envelopes = mod.use_bone_envelopes
            arm.use_deform_preserve_volume = mod.use_deform_preserve_volume


def make_bisected_transfer_source(src_obj, ctx, axes):
    """Create an evaluated, bisected source with export-space coordinates.

    Its vertex group *names* are cloned so deform weights preserved through
    BMesh conversion can be addressed by Blender's weight-transfer modifier.
    The caller owns and removes the returned object and its mesh.
    """
    from . import bisect

    depsgraph = ctx.evaluated_depsgraph_get()
    evaluated = src_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(evaluated_mesh)
        # BMesh's legacy Freestyle layer is not accessible in Blender 4.x.
        # An ordinary integer layer survives the symmetry bisect instead.
        freestyle_layer = bm.edges.layers.int.get("qremeshify_freestyle")
        if freestyle_layer is None:
            freestyle_layer = bm.edges.layers.int.new("qremeshify_freestyle")
        for edge, marked in zip(bm.edges, freestyle_edge_marks(evaluated_mesh)):
            edge[freestyle_layer] = int(marked)
        bmesh.ops.transform(bm, matrix=_rot_scale_matrix(src_obj), verts=bm.verts)
        bisect.bisect_on_axes(bm, *axes)
        mesh = bpy.data.meshes.new("QRemeshify Transfer Source")
        try:
            bm.to_mesh(mesh)
            marks_layer = mesh.attributes.get("qremeshify_freestyle")
            if marks_layer is not None:
                _set_freestyle_marks(mesh, [bool(item.value) for item in marks_layer.data])
                mesh.attributes.remove(marks_layer)
            for material in src_obj.data.materials:
                mesh.materials.append(material)
            mesh.update()
            source = bpy.data.objects.new("QRemeshify Transfer Source", mesh)
            try:
                ctx.collection.objects.link(source)
                source.location = src_obj.location.copy()
                for group in src_obj.vertex_groups:
                    source.vertex_groups.new(name=group.name)
                return source
            except Exception:
                bpy.data.objects.remove(source, do_unlink=True)
                raise
        except Exception:
            bpy.data.meshes.remove(mesh)
            raise
    finally:
        bm.free()
        evaluated.to_mesh_clear()


def _rot_scale_matrix(obj: bpy.types.Object) -> mathutils.Matrix:
    """The same matrix the exporter bakes into the geometry: rotation and scale only."""
    if obj.rotation_mode == "QUATERNION":
        return mathutils.Matrix.LocRotScale(None, obj.rotation_quaternion, obj.scale)
    return mathutils.Matrix.LocRotScale(None, obj.rotation_euler, obj.scale)


def _matched_edges(src_obj: bpy.types.Object, dst_obj: bpy.types.Object):
    """Yield aligned nearest-midpoint source matches for destination edges.

    Shared by UV island constraints and edge-overlay transfer so UV-only mode
    can detect seams without writing seam/sharp attributes to the destination.
    """
    src_mesh, dst_mesh = src_obj.data, dst_obj.data
    if not src_mesh.edges:
        return

    transform = _rot_scale_matrix(src_obj)
    src_coords = [transform @ vertex.co for vertex in src_mesh.vertices]
    tree = KDTree(len(src_mesh.edges))
    directions = {}
    for edge in src_mesh.edges:
        a, b = edge.vertices
        direction = src_coords[b] - src_coords[a]
        if direction.length_squared < 1e-20:
            continue
        direction.normalize()
        directions[edge.index] = direction
        tree.insert((src_coords[a] + src_coords[b]) * 0.5, edge.index)
    if not directions:
        return
    tree.balance()

    dst_verts = dst_mesh.vertices
    for edge in dst_mesh.edges:
        a, b = edge.vertices
        direction = dst_verts[b].co - dst_verts[a].co
        if direction.length_squared < 1e-20:
            continue
        direction.normalize()
        midpoint = (dst_verts[a].co + dst_verts[b].co) * 0.5

        # find_n is distance-ordered. The first aligned candidate is therefore
        # the nearest midpoint *after* excluding edges with |cos| < 0.7.
        count = min(8, len(directions))
        match = None
        while True:
            for _point, index, _distance in tree.find_n(midpoint, count):
                if abs(direction.dot(directions[index])) >= 0.7:
                    match = src_mesh.edges[index]
                    break
            if match is not None or count == len(directions):
                break
            count = min(count * 2, len(directions))

        if match is not None:
            yield edge, match


def _uv_barycentric(point, a, b, c):
    """Weights for a closest point on a source triangle (degenerate-safe)."""
    ab, ac, ap = b - a, c - a, point - a
    d00, d01, d11 = ab.dot(ab), ab.dot(ac), ac.dot(ac)
    d20, d21 = ap.dot(ab), ap.dot(ac)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < 1e-20:
        # Degenerate faces should not normally reach this path.
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return (1.0 - v - w, v, w)


def _uv_corner_outwards(mesh, poly, corner_index, seam_edges):
    """Outward half-planes of seam edges incident to this *face corner*.

    The opposite side of a seam must not be sampled, including at vertices
    shared by three or more UV islands. Each loop gets its own constraints.
    """
    loops = mesh.loops
    vertices = mesh.vertices
    corner = loops[corner_index]
    center = vertices[corner.vertex_index].co
    normal = poly.normal
    directions = []
    for edge_loop_index in (corner_index, poly.loop_start + (corner_index - poly.loop_start - 1) % poly.loop_total):
        edge_loop = loops[edge_loop_index]
        if edge_loop.edge_index not in seam_edges:
            continue
        edge = mesh.edges[edge_loop.edge_index]
        a, b = edge.vertices
        tangent = vertices[b].co - vertices[a].co
        outward = tangent.cross(normal)
        if outward.length_squared < 1e-20:
            continue
        outward.normalize()
        # Choose the direction pointing away from THIS polygon's interior.
        if outward.dot(poly.center - center) > 0:
            outward.negate()
        directions.append(outward)
    return directions
