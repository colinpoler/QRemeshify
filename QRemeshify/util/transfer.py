import bmesh
import bpy
import mathutils
from mathutils.bvhtree import BVHTree


def _rot_scale_matrix(obj: bpy.types.Object) -> mathutils.Matrix:
    """The same matrix the exporter bakes into the geometry: rotation and scale only."""
    if obj.rotation_mode == 'QUATERNION':
        return mathutils.Matrix.LocRotScale(None, obj.rotation_quaternion, obj.scale)
    return mathutils.Matrix.LocRotScale(None, obj.rotation_euler, obj.scale)


def copy_materials(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
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
    # Triangulate so BVH face indices are unambiguous, then remember which
    # material each triangle carries (triangulation preserves material_index)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.faces.ensure_lookup_table()
    material_by_face = [f.material_index for f in bm.faces]
    tree = BVHTree.FromBMesh(bm)

    for poly in dst_mesh.polygons:
        hit = tree.find_nearest(poly.center)
        if hit is not None and hit[2] is not None:
            poly.material_index = material_by_face[hit[2]]

    bm.free()


def transfer_uv_and_weights(src_obj: bpy.types.Object, dst_obj: bpy.types.Object):
    """Transfer UV maps and vertex-group weights via a Data Transfer modifier.

    Works because both objects occupy the same world-space position: the
    remeshed object has rotation/scale baked into its geometry and inherits the
    original's location, so world coordinates line up exactly.

    Returns True if the modifier was applied, False if it had to be left live,
    None if there was nothing to transfer.
    """
    src_mesh = src_obj.data
    has_uv = len(src_mesh.uv_layers) > 0
    has_vg = len(src_obj.vertex_groups) > 0
    if not (has_uv or has_vg):
        return None

    for vg in src_obj.vertex_groups:
        if vg.name not in dst_obj.vertex_groups:
            dst_obj.vertex_groups.new(name=vg.name)
    for uv in src_mesh.uv_layers:
        if uv.name not in dst_obj.data.uv_layers:
            dst_obj.data.uv_layers.new(name=uv.name, do_init=False)

    mod = dst_obj.modifiers.new("QRemeshify Transfer", 'DATA_TRANSFER')
    mod.object = src_obj
    if has_uv:
        mod.use_loop_data = True
        mod.data_types_loops = {'UV'}
        mod.loop_mapping = 'POLYINTERP_NEAREST'
        mod.layers_uv_select_src = 'ALL'
        mod.layers_uv_select_dst = 'NAME'
    if has_vg:
        mod.use_vert_data = True
        mod.data_types_verts = {'VGROUP_WEIGHTS'}
        mod.vert_mapping = 'POLYINTERP_NEAREST'
        mod.layers_vgroup_select_src = 'ALL'
        mod.layers_vgroup_select_dst = 'NAME'

    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return True
    except RuntimeError:
        # Leaving the modifier live still shows correct data in the viewport —
        # better than dropping it on the floor
        return False


def copy_rig(src_obj: bpy.types.Object, dst_obj: bpy.types.Object) -> None:
    """Copy parenting and Armature modifiers so a rigged mesh stays rigged.

    Weights come from transfer_uv_and_weights; this carries over the binding.
    """
    if src_obj.parent is not None:
        dst_obj.parent = src_obj.parent
        dst_obj.parent_type = src_obj.parent_type
        if src_obj.parent_type == 'BONE':
            dst_obj.parent_bone = src_obj.parent_bone
        dst_obj.matrix_parent_inverse = src_obj.matrix_parent_inverse.copy()

    for mod in src_obj.modifiers:
        if mod.type == 'ARMATURE':
            arm = dst_obj.modifiers.new(mod.name, 'ARMATURE')
            arm.object = mod.object
            arm.use_vertex_groups = mod.use_vertex_groups
            arm.use_bone_envelopes = mod.use_bone_envelopes
            arm.use_deform_preserve_volume = mod.use_deform_preserve_volume
