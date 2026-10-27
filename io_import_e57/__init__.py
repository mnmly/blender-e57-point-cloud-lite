"""Import ASTM E57 point clouds as Blender point cloud objects."""

import os

import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, poll_file_object_drop
from mathutils import Quaternion

from .e57 import E57Error, E57File, Scan, spherical_to_cartesian

MATERIAL_NAME = "E57 Point Color"
COLOR_ATTRIBUTE = "Color"
ORIGIN_PROPERTY = "e57_origin"

CARTESIAN = ("cartesianX", "cartesianY", "cartesianZ")
SPHERICAL = ("sphericalRange", "sphericalAzimuth", "sphericalElevation")
COLORS = ("colorRed", "colorGreen", "colorBlue")
NORMALS = ("nor:normalX", "nor:normalY", "nor:normalZ")
# Fields consumed by position/colour/intensity handling, never imported as plain attributes.
CONSUMED = set(CARTESIAN + SPHERICAL + COLORS + NORMALS) | {
    "intensity",
    "cartesianInvalidState",
    "sphericalInvalidState",
}

_INT32 = np.iinfo(np.int32)


class IMPORT_SCENE_OT_e57_lite(bpy.types.Operator, ImportHelper):
    """Import ASTM E57 point cloud scans"""

    bl_idname = "import_scene.e57_lite"
    bl_label = "Import E57"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".e57"
    filter_glob: StringProperty(default="*.e57", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    point_radius: FloatProperty(
        name="Point Radius",
        description="Radius of each point, used for rendering",
        default=0.01,
        min=0.0,
        soft_max=1.0,
        subtype="DISTANCE",
    )
    step: IntProperty(
        name="Keep Every Nth Point",
        description="Subsample large scans by keeping one point in every N",
        default=1,
        min=1,
        soft_max=100,
    )
    import_color: BoolProperty(name="Color", description="Import RGB as a color attribute", default=True)
    import_intensity: BoolProperty(
        name="Intensity", description="Import intensity as a normalized float attribute", default=True
    )
    import_extra: BoolProperty(
        name="Other Fields",
        description="Import remaining per-point fields (row/column, returns, time stamps, normals, "
        "extensions) as attributes",
        default=True,
    )
    skip_invalid: BoolProperty(
        name="Skip Invalid Points",
        description="Drop points whose coordinates are flagged invalid",
        default=True,
    )
    use_pose: BoolProperty(
        name="Apply Scan Pose",
        description="Place each scan with its registered pose (as the object transform)",
        default=True,
    )
    recenter: BoolProperty(
        name="Recenter",
        description="Shift scans so the first imported one sits near the world origin, keeping "
        "float precision for georeferenced data. The shift is stored on the scene "
        f'("{ORIGIN_PROPERTY}") and reused by later imports so scans stay aligned',
        default=True,
    )
    create_material: BoolProperty(
        name="Color Material", description="Assign a material that renders the color attribute", default=True
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "point_radius")
        layout.prop(self, "step")
        col = layout.column(heading="Attributes")
        col.prop(self, "import_color")
        col.prop(self, "import_intensity")
        col.prop(self, "import_extra")
        col = layout.column(heading="Placement")
        col.prop(self, "use_pose")
        col.prop(self, "recenter")
        col = layout.column(heading="Options")
        col.prop(self, "skip_invalid")
        col.prop(self, "create_material")

    def invoke(self, context, event):
        return (
            self.invoke_popup(context) if self.directory or self.filepath else super().invoke(context, event)
        )

    def execute(self, context):
        paths = [os.path.join(self.directory, f.name) for f in self.files if f.name] or [self.filepath]
        wm = context.window_manager
        wm.progress_begin(0, len(paths))
        imported = 0
        try:
            for i, path in enumerate(paths):
                try:
                    imported += self.import_file(context, path, lambda p, i=i: wm.progress_update(i + p))
                except (OSError, E57Error) as e:
                    self.report({"ERROR"}, f"{os.path.basename(path)}: {e}")
        finally:
            wm.progress_end()
        if not imported:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Imported {imported} scan(s) from {len(paths)} file(s)")
        return {"FINISHED"}

    # ------------------------------------------------------------------ import

    def import_file(self, context, path: str, progress) -> int:
        with E57File(path) as f:
            if not f.scans:
                self.report({"WARNING"}, f"{os.path.basename(path)}: no 3D scans in file")
                return 0
            collection = bpy.data.collections.new(os.path.splitext(os.path.basename(path))[0])
            context.collection.children.link(collection)
            count = 0
            for k, scan in enumerate(f.scans):
                obj = self.import_scan(context, f, scan, lambda p, k=k: progress((k + p) / len(f.scans)))
                if obj is not None:
                    collection.objects.link(obj)
                    count += 1
        return count

    def import_scan(self, context, f: E57File, scan: Scan, progress):
        if scan.has(*CARTESIAN):
            position_fields, invalid_field = CARTESIAN, "cartesianInvalidState"
        elif scan.has(*SPHERICAL):
            position_fields, invalid_field = SPHERICAL, "sphericalInvalidState"
        else:
            self.report({"WARNING"}, f"{scan.name}: no cartesian or spherical coordinates, skipped")
            return None

        names = set(position_fields)
        if self.skip_invalid and scan.has(invalid_field):
            names.add(invalid_field)
        if self.import_color and scan.has(*COLORS):
            names.update(COLORS)
        if self.import_intensity and scan.has("intensity"):
            names.add("intensity")
        if self.import_extra:
            names.update(n for n in scan.field_names if n not in CONSUMED)
            if scan.has(*NORMALS):
                names.update(NORMALS)
        data = f.read_points(scan, names, progress)

        keep = np.ones(scan.record_count, bool)
        if invalid_field in data:
            keep &= data.pop(invalid_field) == 0
        if position_fields == SPHERICAL:
            xyz = np.column_stack(spherical_to_cartesian(*(data.pop(n) for n in SPHERICAL)))
        else:
            xyz = np.column_stack([data.pop(n).astype(np.float64) for n in CARTESIAN])
        keep &= np.isfinite(xyz).all(axis=1)
        index = np.flatnonzero(keep)[:: self.step]
        if len(index) == 0:
            self.report({"WARNING"}, f"{scan.name}: no valid points, skipped")
            return None
        xyz = xyz[index]
        data = {k: v[index] for k, v in data.items()}

        # Center the points locally (in float64) and move the offset into the object
        # transform, so float32 positions keep their precision far from the origin.
        # mathutils is float32, so the placement is computed in float64 numpy.
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        xyz -= center
        if self.use_pose:
            rotation = Quaternion(scan.rotation).normalized()
            location = np.array(scan.translation) + scan.rotation_matrix() @ center
        else:
            rotation, location = Quaternion(), center
        if self.recenter:
            location = location - self.scene_origin(context.scene, location)

        pc = bpy.data.pointclouds.new(scan.name)
        pc.resize(len(xyz))
        pc.attributes["position"].data.foreach_set("vector", xyz.astype(np.float32).ravel())
        radius = pc.attributes.new("radius", "FLOAT", "POINT")
        radius.data.foreach_set("value", np.full(len(xyz), self.point_radius, np.float32))

        has_color = all(n in data for n in COLORS)
        if has_color:
            rgb = [self.normalize(scan, n, data.pop(n), scan.color_limits.get(n)) for n in COLORS]
            rgba = np.column_stack(rgb + [np.ones(len(xyz))]).astype(np.float32)
            color = pc.attributes.new(COLOR_ATTRIBUTE, "BYTE_COLOR", "POINT")
            color.data.foreach_set("color_srgb", rgba.ravel())
            pc.color_attributes.active = color
        if "intensity" in data:
            values = self.normalize(scan, "intensity", data.pop("intensity"), scan.intensity_limits)
            pc.attributes.new("intensity", "FLOAT", "POINT").data.foreach_set(
                "value", values.astype(np.float32)
            )
        if all(n in data for n in NORMALS):
            normals = np.column_stack([data.pop(n) for n in NORMALS]).astype(np.float32)
            pc.attributes.new("normal", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", normals.ravel())
        for name, values in data.items():
            self.add_attribute(pc, name, values)

        if self.create_material and has_color:
            pc.materials.append(color_material())

        obj = bpy.data.objects.new(scan.name, pc)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation
        obj.location = location
        obj["e57_guid"] = scan.guid
        return obj

    @staticmethod
    def normalize(scan: Scan, name: str, values: np.ndarray, limits) -> np.ndarray:
        """Map values to 0..1 using the scan's limits, else the field's declared range."""
        if limits is None:
            spec = next(s for s in scan.fields if s.name == name)
            if spec.type == "Float" or spec.maximum <= spec.minimum:
                limits = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
            else:
                limits = spec.minimum * spec.scale + spec.offset, spec.maximum * spec.scale + spec.offset
        lo, hi = limits
        if hi <= lo:
            return np.zeros(len(values))
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def add_attribute(pc, name: str, values: np.ndarray):
        name = name.replace(":", "_")
        if pc.attributes.get(name) is not None:
            return
        fits_int32 = len(values) == 0 or (values.min() >= _INT32.min and values.max() <= _INT32.max)
        if values.dtype.kind == "i" and fits_int32:
            pc.attributes.new(name, "INT", "POINT").data.foreach_set("value", values.astype(np.int32))
        else:
            pc.attributes.new(name, "FLOAT", "POINT").data.foreach_set("value", values.astype(np.float32))

    @staticmethod
    def scene_origin(scene, first_location: np.ndarray) -> np.ndarray:
        if ORIGIN_PROPERTY not in scene:
            scene[ORIGIN_PROPERTY] = [float(v) for v in first_location]  # stored as doubles
        return np.array(scene[ORIGIN_PROPERTY], np.float64)


def color_material():
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is not None:
        return mat
    # Scan colors already contain the scene lighting, so render them unlit.
    mat = bpy.data.materials.new(MATERIAL_NAME)
    tree = mat.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    attr = tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = COLOR_ATTRIBUTE
    emission.location = (-200, 0)
    attr.location = (-400, 0)
    tree.links.new(attr.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


class IO_FH_e57_lite(bpy.types.FileHandler):
    bl_idname = "IO_FH_e57_lite"
    bl_label = "E57"
    bl_import_operator = "import_scene.e57_lite"
    bl_file_extensions = ".e57"

    @classmethod
    def poll_drop(cls, context):
        return poll_file_object_drop(context)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_e57_lite.bl_idname, text="E57 Point Cloud (.e57)")


classes = (IMPORT_SCENE_OT_e57_lite, IO_FH_e57_lite)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
