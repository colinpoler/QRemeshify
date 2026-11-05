import os

import bpy
import bmesh
import mathutils

from .lib import Quadwild, QWException
from .util import bisect, exporter, importer, transfer, density


class QREMESH_OT_Remesh(bpy.types.Operator):
    """Remesh with Quadwild"""
    bl_idname = "qremeshify.remesh"
    bl_label = "Remesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'OBJECT'

    def execute(self, ctx):
        props = ctx.scene.quadwild_props
        qr_props = ctx.scene.quadpatches_props
        selected_objs = ctx.selected_objects

        if len(selected_objs) == 0:
            self.report({'ERROR_INVALID_INPUT'}, "No selected objects")
            return {'CANCELLED'}

        if len(selected_objs) > 1:
            self.report({'INFO'}, "Multiple objects selected, will only operate on the first selected object")

        obj = selected_objs[0]
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR_INVALID_INPUT'}, "Object is not a mesh")
            return {'CANCELLED'}

        if len(obj.data.polygons) == 0:
            self.report({'ERROR_INVALID_INPUT'}, "Mesh has 0 faces")
            return {'CANCELLED'}

        original_location = obj.location.copy()

        mesh_filename = "".join(c if c not in "\\/:*?<>|" else "_" for c in obj.name).strip()
        mesh_filepath = f"{os.path.join(bpy.app.tempdir, mesh_filename)}.obj"
        self.report({'DEBUG'}, f"Remeshing from {mesh_filepath}")

        # The solver libraries run inside Blender's process: a native crash takes
        # the whole application down with it. A backup copy is cheap insurance.
        if bpy.data.filepath:
            if bpy.data.is_dirty:
                backup_path = f"{os.path.splitext(bpy.data.filepath)[0]}_qremesh_bak.blend"
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=backup_path, copy=True)
                    self.report({'INFO'}, f"Backup saved to {backup_path}")
                except RuntimeError as e:
                    self.report({'WARNING'}, f"Could not save backup: {e}")
        else:
            self.report({'WARNING'}, "File has never been saved — a crash during remesh would lose your work")

        # Load lib
        qw = Quadwild(mesh_filepath)

        # Use Cache skips straight to quadrangulation, which reads files produced
        # by a previous full run. Handing missing paths to the solver crashes
        # Blender, so refuse early with a clear message instead.
        if props.useCache and not os.path.isfile(qw.traced_path):
            self.report({'ERROR_INVALID_INPUT'},
                        f"No cached files found for '{obj.name}' — run once with Use Cache disabled first")
            return {'CANCELLED'}

        bm = None
        evaluated_obj = None
        try:
            if not props.useCache:
                # Get mesh after modifiers and shapekeys applied
                depsgraph = bpy.context.evaluated_depsgraph_get()
                evaluated_obj = obj.evaluated_get(depsgraph)
                mesh = evaluated_obj.to_mesh()

                # Create a bmesh from mesh
                # (won't affect mesh, unless explicitly written back)
                bm = bmesh.new()
                bm.from_mesh(mesh)
                # Freestyle is not exposed reliably by BMesh's legacy edge API.
                # Preserve marks in an integer edge layer across bisect/decimate.
                freestyle_layer = bm.edges.layers.int.get("qremeshify_freestyle")
                if freestyle_layer is None:
                    freestyle_layer = bm.edges.layers.int.new("qremeshify_freestyle")
                for edge, marked in zip(bm.edges, transfer.freestyle_edge_marks(mesh)):
                    edge[freestyle_layer] = int(marked)
                    if marked and props.enableSharp:
                        edge.smooth = False  # Treat Freestyle as a QuadWild feature.

                # Apply only rotation and scale
                if evaluated_obj.rotation_mode == 'QUATERNION':
                    matrix = mathutils.Matrix.LocRotScale(None, evaluated_obj.rotation_quaternion, evaluated_obj.scale)
                else:
                    matrix = mathutils.Matrix.LocRotScale(None, evaluated_obj.rotation_euler, evaluated_obj.scale)
                bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)

                # Bisect to prep for symmetry
                if props.symmetryX or props.symmetryY or props.symmetryZ:
                    bisect.bisect_on_axes(bm, props.symmetryX, props.symmetryY, props.symmetryZ)

                # Mark feature edges and select their vertices for protected decimation.
                bm.verts.index_update()
                face_set_layer = bm.faces.layers.int.get('.sculpt_face_set')
                freestyle_layer = bm.edges.layers.int.get("qremeshify_freestyle")
                sharp_vertices = set()
                for edge in bm.edges:
                    if exporter.is_sharp_feature(edge, props.sharpAngle, face_set_layer, freestyle_layer):
                        edge.select_set(True)
                        for vert in edge.verts:
                            vert.select_set(True)
                            sharp_vertices.add(vert.index)
                        if props.enableSharp:
                            edge.smooth = False

                if props.enableRemesh:
                    # Decimate a temporary Blender object, never the user's source mesh.
                    prep_mesh = bpy.data.meshes.new("QRemeshify Preprocess")
                    prep_obj = None
                    try:
                        bm.to_mesh(prep_mesh)
                        prep_obj = bpy.data.objects.new("QRemeshify Preprocess", prep_mesh)
                        ctx.collection.objects.link(prep_obj)
                        sharp_group = prep_obj.vertex_groups.new(name="qremeshify_sharp_vertices")
                        if sharp_vertices:
                            sharp_group.add(sorted(sharp_vertices), 1.0, 'REPLACE')
                        decimate = prep_obj.modifiers.new("QRemeshify Decimate", 'DECIMATE')
                        decimate.decimate_type = 'COLLAPSE'
                        decimate.ratio = 0.5
                        decimate.vertex_group = sharp_group.name
                        decimate.invert_vertex_group = True
                        prep_eval = prep_obj.evaluated_get(depsgraph)
                        prep_eval_mesh = prep_eval.to_mesh()
                        try:
                            bm.clear()
                            bm.from_mesh(prep_eval_mesh)
                        finally:
                            prep_eval.to_mesh_clear()
                    finally:
                        if prep_obj is not None:
                            bpy.data.objects.remove(prep_obj, do_unlink=True)
                        bpy.data.meshes.remove(prep_mesh)

                # Triangulate mesh
                bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='SHORT_EDGE', ngon_method='BEAUTY')
                bm.verts.index_update()

                # Export selected object as OBJ
                exporter.export_mesh(bm, mesh_filepath)

                # Calculate sharp features
                if props.enableSharp:
                    num_sharp_features = exporter.export_sharp_features(bm, qw.sharp_path, props.sharpAngle)
                    self.report({'DEBUG'}, f"Found {num_sharp_features} sharp edges")

                # Remesh and calculate field
                qw.remeshAndField(enableSharp=props.enableSharp, sharpAngle=props.sharpAngle)
                if props.debug:
                    new_mesh = importer.import_mesh(qw.remeshed_path)
                    new_obj = bpy.data.objects.new(f"{obj.name} remeshAndField", new_mesh)
                    bpy.context.collection.objects.link(new_obj)
                    new_obj.hide_set(True)

                # Trace
                qw.trace()
                if props.debug:
                    new_mesh = importer.import_mesh(qw.traced_path)
                    new_obj = bpy.data.objects.new(f"{obj.name} trace", new_mesh)
                    bpy.context.collection.objects.link(new_obj)
                    new_obj.hide_set(True)

            # QuadWild patch IDs are only available after tracing. Refresh the
            # scales even on cache hits, so repainting density needs no retrace.
            density_count = density.write_patch_scales(
                qw.traced_path, obj, ctx.evaluated_depsgraph_get(),
                transfer._rot_scale_matrix(obj),
                (props.symmetryX, props.symmetryY, props.symmetryZ))
            if density_count:
                self.report({'INFO'}, f"Painted density applied to {density_count} patches")

            # Convert to quads
            solver_result = qw.quadrangulate(
                props.enableSmoothing,
                qr_props.scaleFact,
                qr_props.fixedChartClusters,

                qr_props.alpha,
                qr_props.ilpMethod,
                qr_props.timeLimit,
                qr_props.gapLimit,
                qr_props.minimumGap,
                qr_props.isometry,
                qr_props.regularityQuadrilaterals,
                qr_props.regularityNonQuadrilaterals,
                qr_props.regularityNonQuadrilateralsWeight,
                qr_props.alignSingularities,
                qr_props.alignSingularitiesWeight,
                qr_props.repeatLosingConstraintsIterations,
                qr_props.repeatLosingConstraintsQuads,
                qr_props.repeatLosingConstraintsNonQuads,
                qr_props.repeatLosingConstraintsAlign,
                qr_props.hardParityConstraint,

                qr_props.flowConfig,
                qr_props.satsumaConfig,

                qr_props.callbackTimeLimit,
                qr_props.callbackGapLimit,
            )
            if props.debug and props.enableSmoothing:
                new_mesh = importer.import_mesh(qw.output_path)
                new_obj = bpy.data.objects.new(f"{obj.name} quadrangulate", new_mesh)
                bpy.context.collection.objects.link(new_obj)
                new_obj.hide_set(True)

            # Import final OBJ
            final_mesh_path = qw.output_smoothed_path if props.enableSmoothing else qw.output_path
            if not os.path.isfile(final_mesh_path):
                raise QWException(
                    f"Quadrangulation produced no output (solver returned {solver_result}) — "
                    "try different settings, or disable Use Cache if the mesh has changed")
            final_mesh = importer.import_mesh(final_mesh_path)
            final_obj = bpy.data.objects.new(f"{obj.name} Remeshed", final_mesh)
            bpy.context.collection.objects.link(final_obj)
            bpy.context.view_layer.objects.active = final_obj
            final_obj.select_set(True)

            # Set object location
            final_obj.location = original_location

            # Transfer against the evaluated, bisected half before mirroring;
            # never match attributes against the opposite symmetry half.
            if props.transferData and any((props.keepUV, props.keepVertexGroups,
                                           props.keepEdgeOverlays, props.keepMaterials)):
                axes = (props.symmetryX, props.symmetryY, props.symmetryZ)
                transfer_source = None
                try:
                    if any(axes):
                        transfer_source = transfer.make_bisected_transfer_source(obj, ctx, axes)
                    source = transfer_source if transfer_source is not None else obj
                    # Each data category is independent and runs on the
                    # unmirrored result using only the (possibly bisected) source.
                    if props.keepMaterials:
                        transfer.transfer_materials(source, final_obj)
                    if props.keepVertexGroups:
                        if not transfer.transfer_vertex_groups(source, final_obj):
                            self.report({'WARNING'}, "Could not apply vertex-group Data Transfer modifier")
                        transfer.copy_rig(obj, final_obj)
                    if props.keepEdgeOverlays:
                        transfer.transfer_edge_overlays(source, final_obj)
                    if props.keepUV:
                        transfer.transfer_uvs(source, final_obj)
                finally:
                    if transfer_source is not None:
                        source_mesh = transfer_source.data
                        bpy.data.objects.remove(transfer_source, do_unlink=True)
                        bpy.data.meshes.remove(source_mesh)

            # Add Mirror only after all transfers have completed. Leave it live.
            if props.symmetryX or props.symmetryY or props.symmetryZ:
                mirror_modifier = final_obj.modifiers.new("Mirror", "MIRROR")
                mirror_modifier.use_axis[0] = props.symmetryX
                mirror_modifier.use_axis[1] = props.symmetryY
                mirror_modifier.use_axis[2] = props.symmetryZ
                mirror_modifier.use_clip = True
                mirror_modifier.merge_threshold = 0.001

            # Hide original
            obj.hide_set(True)

        except QWException as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Unexpected error: {e!r}")
            return {'CANCELLED'}
        finally:
            # Cleanup
            del qw

            if bm is not None:
                bm.free()
            if evaluated_obj is not None:
                evaluated_obj.to_mesh_clear()

        return {'FINISHED'}
