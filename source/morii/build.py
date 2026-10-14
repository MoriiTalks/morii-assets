"""Build Morii's editable rig, runtime GLB, and visual review frames in Blender."""

import argparse
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "exports" / "models"
PREVIEW = ROOT / "exports" / "images"
REVIEW = ROOT / "render-output" / "morii"
FPS = 24
PARTS = []
CLIPS = {
    "idle": 4.0,
    "wave": 2.4,
    "listen": 2.4,
    "think": 2.8,
    "speak": 2.4,
    "encourage": 2.4,
    "move_left": 3.0,
    "move_right": 3.0,
    "move_forward": 3.0,
    "move_backward": 3.0,
    "reset": 0.6,
}


def material(name, color, roughness=0.7):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = next(node for node in result.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Specular IOR Level"].default_value = 0.3
    return result


def finish_mesh(obj, name, surface, bone, mask=None):
    obj.name = name
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.materials.append(surface)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    group = obj.vertex_groups.new(name=bone)
    group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
    if mask:
        obj.vertex_groups.new(name=mask).add(list(range(len(obj.data.vertices))), 1, "REPLACE")
    PARTS.append(obj)
    return obj


def ellipsoid(name, center, scale, surface, bone, segments=24, rings=16, mask=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=center)
    obj = bpy.context.object
    obj.scale = scale
    return finish_mesh(obj, name, surface, bone, mask)


def capsule(name, start, end, width, depth, surface, bone):
    direction = Vector(end) - Vector(start)
    center = (Vector(start) + Vector(end)) / 2
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, location=center)
    obj = bpy.context.object
    obj.scale = (width, depth, direction.length / 2 + width * 0.65)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    return finish_mesh(obj, name, surface, bone)


def oriented_ellipsoid(name, center, scale, surface, bone, rotation, segments=20, rings=12):
    obj = ellipsoid(name, center, scale, surface, bone, segments, rings)
    origin = Vector(center)
    for vertex in obj.data.vertices:
        vertex.co = origin + rotation @ (vertex.co - origin)
    return obj


def line(name, points, radius, surface, bone, radii=None):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 8
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for index, (point, coordinate) in enumerate(zip(spline.bezier_points, points)):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
        if radii:
            point.radius = radii[index]
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    return finish_mesh(bpy.context.object, name, surface, bone)


def tail(surface, cream):
    vertices = []
    faces = []
    radial = 20
    rings = 32
    for ring in range(rings):
        t = ring / (rings - 1)
        path_t = min(t, 0.84)
        center = Vector((-0.14 - 1.08 * path_t, 0.22 + 0.16 * math.sin(path_t * math.pi), 0.48 + 0.40 * path_t ** 3))
        tangent = Vector((-1.08, 0.16 * math.pi * math.cos(path_t * math.pi), 1.20 * path_t ** 2)).normalized()
        side = tangent.cross(Vector((0, 0, 1))).normalized()
        vertical = side.cross(tangent).normalized()
        radius = 0.245 * (1 - 0.28 * t) * math.sin(math.pi * min(1, t / 0.09) / 2)
        if t > 0.84:
            cap = (t - 0.84) / 0.16
            center += tangent * (0.187 * cap)
            radius = 0.187 * math.sqrt(max(0, 1 - cap ** 2))
        if ring == rings - 1:
            radius = 0.002
        for segment in range(radial):
            angle = segment * math.tau / radial
            vertices.append(center + radius * (side * math.cos(angle) + vertical * math.sin(angle)))
        if ring:
            for segment in range(radial):
                next_segment = (segment + 1) % radial
                faces.append(((ring - 1) * radial + segment, (ring - 1) * radial + next_segment, ring * radial + next_segment, ring * radial + segment))
    mesh = bpy.data.meshes.new("TailGeometry")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("TailShape", mesh)
    bpy.context.collection.objects.link(obj)
    finish_mesh(obj, "TailShape", surface, "Tail")
    obj.data.materials.append(cream)
    for polygon in mesh.polygons:
        segment = polygon.index % radial
        polygon.material_index = 1 if polygon.index // radial < rings - 4 and math.sin((segment + 0.5) * math.tau / radial) < -0.45 else 0
    tip_group = obj.vertex_groups.new(name="TailTip")
    root_group = obj.vertex_groups["Tail"]
    for vertex in mesh.vertices:
        t = min(1, max(0, (-vertex.co.x - 0.3) / 0.65))
        tip_group.add([vertex.index], t, "REPLACE")
        root_group.add([vertex.index], 1 - t, "REPLACE")


def muzzle_front(x, z):
    radius = max(0, 1 - (x / 0.805) ** 2 - ((z - 1.985) / 0.36) ** 2)
    return -0.414 - 0.272 * math.sqrt(radius)


def blush_patch(name, x, z, surface):
    vertices = [(x, muzzle_front(x, z) - 0.003, z)]
    faces = []
    radial = 24
    for ring in range(1, 4):
        for segment in range(radial):
            angle = segment * math.tau / radial
            vx = x + 0.099 * ring / 3 * math.cos(angle)
            vz = z + 0.074 * ring / 3 * math.sin(angle)
            vertices.append((vx, muzzle_front(vx, vz) - 0.003, vz))
            current = 1 + (ring - 1) * radial + segment
            following = 1 + (ring - 1) * radial + (segment + 1) % radial
            if ring == 1:
                faces.append((0, current, following))
            else:
                previous = current - radial
                previous_next = following - radial
                faces.append((previous, current, following, previous_next))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return finish_mesh(obj, name, surface, "Head")


def chest_patch(body, surface):
    # Project the cream marking onto the body so the neck and belly share its contour.
    tree = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    vertices = []
    faces = []
    weights = []
    radial, rings = 64, 24
    coordinates = [(0, 1.16)]
    for ring in range(1, rings + 1):
        for segment in range(radial):
            angle = segment * math.tau / radial
            z = 1.16 + 0.78 * ring / rings * math.sin(angle)
            taper = 1 - 0.28 * max(0, (z - 1.05) / 0.90)
            x = 0.41 * ring / rings * math.cos(angle) * taper
            coordinates.append((x, z))
    for x, z in coordinates:
        hit, normal, face_index, _distance = tree.ray_cast(Vector((x, -2, z)), Vector((0, 1, 0)))
        if hit is None:
            raise ValueError("The cream marking extends beyond the body")
        vertices.append(hit + normal * 0.008)
        polygon = body.data.polygons[face_index]
        nearest = min(polygon.vertices, key=lambda index: (body.data.vertices[index].co - hit).length)
        weights.append([(group.group, group.weight) for group in body.data.vertices[nearest].groups])
    for ring in range(1, rings + 1):
        for segment in range(radial):
            current = 1 + (ring - 1) * radial + segment
            following = 1 + (ring - 1) * radial + (segment + 1) % radial
            if ring == 1:
                faces.append((0, current, following))
            else:
                faces.append((current - radial, current, following, following - radial))
    mesh = bpy.data.meshes.new("ChestMarking")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("ChestMarking", mesh)
    bpy.context.collection.objects.link(obj)
    finish_mesh(obj, "ChestMarking", surface, "Spine")
    obj.vertex_groups.clear()
    groups = [obj.vertex_groups.new(name=group.name) for group in body.vertex_groups]
    for index, influences in enumerate(weights):
        for group, weight in influences:
            groups[group].add([index], weight, "REPLACE")


def arm_points(sign):
    return (
        (sign * 0.394, 0.025, 1.54),
        (sign * 0.571, -0.045, 1.25),
        (sign * 0.444, -0.239, 1.105),
        (sign * 0.410, -0.285, 1.045),
        (sign * 0.328, -0.305, 1.02),
    )


def make_character(surfaces):
    teal, cream, peach, brown, white, eye, paper = surfaces
    body = ellipsoid("Body", (0, 0.06, 1.07), (0.59, 0.37, 0.73), teal, "Spine", 36, 24)
    for vertex in body.data.vertices:
        waist = 1 - 0.22 * max(0, (vertex.co.z - 1.08) / 0.60)
        vertex.co.x *= waist
    ellipsoid("Neck", (0, 0.01, 1.67), (0.28, 0.25, 0.29), teal, "Spine")
    head = ellipsoid("Head", (0, -0.005, 2.20), (0.848, 0.525, 0.69), teal, "Head", 40, 28)
    for vertex in head.data.vertices:
        vertex.co.x *= 1 - 0.18 * max(0, (vertex.co.z - 2.19) / 0.69)
    ellipsoid("Muzzle", (0, -0.414, 1.985), (0.805, 0.272, 0.36), cream, "Head", 40, 26)

    for sign, suffix in ((1, "L"), (-1, "R")):
        ellipsoid("Ear." + suffix, (sign * 0.704, 0.048, 2.56), (0.19, 0.105, 0.207), teal, "Head")
        ellipsoid("EarInner." + suffix, (sign * 0.713, -0.055, 2.57), (0.118, 0.025, 0.134), cream, "Head", 20, 12)
        blush_patch("Cheek." + suffix, sign * 0.578, 2.091, peach)
        ellipsoid("EyeWhite." + suffix, (sign * 0.437, -0.534, 2.284), (0.104, 0.038, 0.114), white, "Head", 24, 16, "EyeBlink")
        ellipsoid("Eye." + suffix, (sign * 0.435, -0.577, 2.288), (0.100, 0.054, 0.108), eye, "Head", 24, 16, "EyeBlink")
        ellipsoid("EyeLight." + suffix, (sign * 0.435 - 0.027, -0.624, 2.327), (0.021, 0.005, 0.021), white, "Head", 16, 10, "EyeBlink")
        line("Brow." + suffix, [(sign * 0.368, -0.425, 2.492), (sign * 0.423, -0.421, 2.548), (sign * 0.488, -0.391, 2.512)], 0.018, brown, "Head", (0.3, 1, 0.65))
        for index in range(3):
            z = 2.032 - index * 0.069
            line("Whisker." + suffix + str(index), [(sign * 0.665, -0.549, z), (sign * 0.750, -0.545, z - 0.015 - index * 0.016), (sign * 0.816, -0.511, z - 0.015 - index * 0.043)], 0.009, brown, "Head")
        shoulder, elbow, wrist, hand, tip = arm_points(sign)
        axis = (Vector(tip) - Vector(wrist)).normalized()
        palm_normal = (Vector((0, 1, 0)) - axis * axis.y).normalized()
        side = palm_normal.cross(axis).normalized()
        hand_rotation = Matrix((side, palm_normal, axis)).transposed().to_quaternion()
        capsule("UpperArm." + suffix, shoulder, elbow, 0.165, 0.145, teal, "Arm." + suffix)
        capsule("Forearm." + suffix, elbow, wrist, 0.157, 0.146, teal, "Forearm." + suffix)
        oriented_ellipsoid("Hand." + suffix, hand, (0.155, 0.125, 0.185), teal, "Hand." + suffix, hand_rotation, 24, 16)
        palm = Vector(hand) + palm_normal * 0.125 - axis * 0.025
        oriented_ellipsoid("Palm." + suffix, palm, (0.088, 0.012, 0.082), cream, "Hand." + suffix, hand_rotation)
        for digit in range(3):
            center = Vector(hand) + axis * (0.12 + (0.025 if digit == 1 else 0)) + side * ((digit - 1) * 0.090)
            oriented_ellipsoid("Finger." + suffix + str(digit), center, (0.059, 0.103, 0.075), teal, "Hand." + suffix, hand_rotation, 16, 10)
            oriented_ellipsoid("Pad." + suffix + str(digit), center + palm_normal * 0.106, (0.028, 0.010, 0.028), cream, "Hand." + suffix, hand_rotation, 16, 10)
        ellipsoid("Leg." + suffix, (sign * 0.367, 0.033, 0.38), (0.282, 0.268, 0.325), teal, "Leg." + suffix)
        ellipsoid("Foot." + suffix, (sign * 0.425, -0.056, 0.147), (0.265, 0.275, 0.14), teal, "Foot." + suffix)
        for digit in range(3):
            ellipsoid("Toe." + suffix + str(digit), (sign * (0.270 + digit * 0.155), -0.232, 0.115), (0.102, 0.124, 0.106), teal, "Foot." + suffix, 20, 12)

    for index, coordinate in enumerate(((-0.12, 0.022, 2.885), (0.015, 0.024, 2.861))):
        obj = ellipsoid("Tuft" + str(index), coordinate, (0.067, 0.065, 0.142 - index * 0.031), teal, "Head", 20, 12)
        for vertex in obj.data.vertices:
            vertex.co.x += max(0, vertex.co.z - 2.79) * 0.51
    nose = ellipsoid("Nose", (0, -0.697, 2.173), (0.144, 0.074, 0.094), brown, "Head", 28, 18)
    for vertex in nose.data.vertices:
        vertex.co.x *= 0.94 + 0.30 * (vertex.co.z - 2.173) / 0.094
    line("NoseLine", [(0, -0.758, 2.124), (0, -0.718, 2.054), (0, -0.702, 2.019)], 0.0105, brown, "Head")
    for sign in (-1, 1):
        smile = [(0, 2.019), (sign * 0.124, 1.970), (sign * 0.267, 1.991), (sign * 0.312, 2.063)]
        line("Smile" + str(sign), [(x, muzzle_front(x, z) - 0.008, z) for x, z in smile], 0.0105, brown, "Head")
        ellipsoid("SmileCorner" + str(sign), (sign * 0.310, muzzle_front(sign * 0.310, 2.061) - 0.006, 2.061), (0.012, 0.009, 0.012), brown, "Head", 12, 8)
    ellipsoid("MouthInner", (0, -0.704, 2.007), (0.081, 0.009, 0.0025), brown, "Head", 24, 12, "MouthOpen")
    tail(teal, cream)

    # A single rounded body surface removes primitive joints while keeping a small rig.
    fuse_body(teal)
    chest_patch(body, cream)
    fuse_arm(teal, "L")
    fuse_arm(teal, "R")

    bpy.ops.object.select_all(action="DESELECT")
    for part in PARTS:
        part.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    body.name = "MoriiBody"
    body.data.name = "MoriiGeometry"
    body.shape_key_add(name="Basis")
    blink = body.shape_key_add(name="blink")
    mouth = body.shape_key_add(name="mouthOpen")
    blink.value = 0
    mouth.value = 0
    eye_group = body.vertex_groups["EyeBlink"].index
    mouth_group = body.vertex_groups["MouthOpen"].index
    for vertex in body.data.vertices:
        groups = {group.group for group in vertex.groups}
        if eye_group in groups:
            blink.data[vertex.index].co.z = 2.288 + (vertex.co.z - 2.288) * 0.075
        if mouth_group in groups:
            mouth.data[vertex.index].co.z = 1.994 + (vertex.co.z - 2.007) * 18
    head_group = body.vertex_groups["Head"].index
    head_center = Vector((0, 0, 2.20))
    head_roll = Quaternion((0, 1, 0), math.radians(-8))
    for key in body.data.shape_keys.key_blocks:
        for vertex in body.data.vertices:
            if any(group.group == head_group for group in vertex.groups):
                key.data[vertex.index].co = head_center + head_roll @ (key.data[vertex.index].co - head_center)
    body.vertex_groups.remove(body.vertex_groups["EyeBlink"])
    body.vertex_groups.remove(body.vertex_groups["MouthOpen"])
    body["character"] = "Morii"
    body["morphs"] = "blink,mouthOpen"
    return body


def fuse_body(surface):
    pieces = [obj for obj in PARTS if obj.name in ("Body", "Neck") or obj.name.startswith("Leg.")]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in pieces:
        obj.select_set(True)
        PARTS.remove(obj)
    bpy.context.view_layer.objects.active = pieces[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = "BodyCore"
    obj.vertex_groups.clear()
    remesh = obj.modifiers.new("ContinuousSurface", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.038
    remesh.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    smooth = obj.modifiers.new("SoftSurface", "SMOOTH")
    smooth.factor = 0.7
    smooth.iterations = 4
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    decimate = obj.modifiers.new("RuntimeBudget", "DECIMATE")
    decimate.ratio = 0.64
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    obj.data.materials.clear()
    obj.data.materials.append(surface)
    joints = {}
    for sign, suffix in ((1, "L"), (-1, "R")):
        joints["Arm." + suffix] = Vector((sign * 0.48, 0.008, 1.36))
        joints["Forearm." + suffix] = Vector((sign * 0.575, -0.097, 1.155))
        joints["Hand." + suffix] = Vector((sign * 0.587, -0.2, 1.00))
        joints["Leg." + suffix] = Vector((sign * 0.367, 0.033, 0.38))
        joints["Foot." + suffix] = Vector((sign * 0.425, -0.19, 0.14))
    groups = {name: obj.vertex_groups.new(name=name) for name in ("Spine", *joints.keys())}
    for vertex in obj.data.vertices:
        x, y, z = vertex.co
        suffix = "L" if x > 0 else "R"
        if z < 0.55:
            names = ("Leg." + suffix, "Foot." + suffix)
            weights = [1 / max(0.04, (vertex.co - joints[name]).length) ** 4 for name in names]
            if z > 0.4 and abs(x) < 0.21:
                names = (*names, "Spine")
                weights.append(((z - 0.4) / 0.15) * sum(weights))
        else:
            names, weights = ("Spine",), [1]
        total = sum(weights)
        for name, weight in zip(names, weights):
            if weight > 0:
                groups[name].add([vertex.index], weight / total, "REPLACE")
        vertex.co.z = max(0.016, z)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    PARTS.append(obj)


def fuse_arm(surface, suffix):
    prefixes = tuple(name + suffix for name in ("UpperArm.", "Forearm.", "Hand.", "Finger."))
    pieces = [obj for obj in PARTS if obj.name.startswith(prefixes)]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in pieces:
        obj.select_set(True)
        PARTS.remove(obj)
    bpy.context.view_layer.objects.active = pieces[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = "ArmSurface." + suffix
    obj.vertex_groups.clear()
    remesh = obj.modifiers.new("ContinuousArm", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = 0.028
    remesh.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    smooth = obj.modifiers.new("SoftArm", "SMOOTH")
    smooth.factor = 0.65
    smooth.iterations = 3
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    decimate = obj.modifiers.new("ArmBudget", "DECIMATE")
    decimate.ratio = 0.65
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    obj.data.materials.clear()
    obj.data.materials.append(surface)
    sign = 1 if suffix == "L" else -1
    shoulder, elbow, wrist, hand, tip = (Vector(point) for point in arm_points(sign))
    centers = {
        "Arm." + suffix: (shoulder + elbow) / 2,
        "Forearm." + suffix: (elbow + wrist) / 2,
        "Hand." + suffix: (hand + tip) / 2,
    }
    groups = {name: obj.vertex_groups.new(name=name) for name in centers}
    hand_axis = (tip - wrist).normalized()
    hand_length = (tip - wrist).length
    for vertex in obj.data.vertices:
        weights = {name: 1 / max(0.045, (vertex.co - center).length) ** 5 for name, center in centers.items()}
        total = sum(weights.values())
        # The palm and fingers turn with the wrist, the forearm still blends into the upper arm.
        amount = min(1, max(0, (vertex.co - wrist).dot(hand_axis) / hand_length / 0.35))
        amount = amount * amount * (3 - 2 * amount)
        for name, weight in weights.items():
            influence = weight / total * (1 - amount) + (amount if name == "Hand." + suffix else 0)
            groups[name].add([vertex.index], influence, "REPLACE")
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    PARTS.append(obj)


def make_rig(body):
    armature = bpy.data.armatures.new("MoriiSkeleton")
    rig = bpy.data.objects.new("MoriiRig", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    definitions = [
        ("Root", (0, 0, 0), (0, 0, 0.2), None),
        ("Spine", (0, 0.035, 0.43), (0, 0.035, 1.66), "Root"),
        ("Head", (0, 0.018, 1.68), (0, 0.018, 2.53), "Spine"),
        ("Tail", (-0.12, 0.22, 0.48), (-0.67, 0.41, 0.59), "Root"),
        ("TailTip", (-0.67, 0.41, 0.59), (-1.18, 0.22, 0.85), "Tail"),
    ]
    for sign, suffix in ((1, "L"), (-1, "R")):
        shoulder, elbow, wrist, hand, tip = arm_points(sign)
        definitions += [
            ("Arm." + suffix, shoulder, elbow, "Spine"),
            ("Forearm." + suffix, elbow, wrist, "Arm." + suffix),
            ("Hand." + suffix, wrist, tip, "Forearm." + suffix),
            ("Leg." + suffix, (sign * 0.367, 0.033, 0.53), (sign * 0.425, 0, 0.22), "Root"),
            ("Foot." + suffix, (sign * 0.425, 0, 0.22), (sign * 0.425, -0.29, 0.12), "Leg." + suffix),
        ]
    for name, head, end, parent in definitions:
        bone = armature.edit_bones.new(name)
        bone.head = head
        bone.tail = end
        if parent:
            bone.parent = armature.edit_bones[parent]
        bone.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
    body.parent = rig
    modifier = body.modifiers.new("MoriiSkin", "ARMATURE")
    modifier.object = rig
    rig.show_in_front = True
    return rig


def rotate(rig, name, axis, angle):
    bone = rig.pose.bones[name]
    rest = bone.bone.matrix_local.to_quaternion()
    bone.rotation_quaternion = rest.inverted() @ Quaternion(axis, angle) @ rest


def pose(rig, clip, t):
    for bone in rig.pose.bones:
        bone.rotation_quaternion = Quaternion()
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)
    envelope = math.sin(math.pi * t) ** 2
    breath = math.sin(t * math.tau) * envelope
    if clip == "reset":
        return
    rotate(rig, "Head", (0, 1, 0), 0.035 * breath)
    rotate(rig, "Tail", (0, 0, 1), 0.08 * breath)
    rotate(rig, "TailTip", (0, 0, 1), 0.045 * math.sin(t * math.tau + 0.5) * envelope)
    rig.pose.bones["Spine"].scale = (1 + 0.009 * breath, 1 + 0.009 * breath, 1 + 0.007 * breath)
    if clip == "wave":
        rotate(rig, "Arm.L", (0, 1, 0), -1.20 * envelope)
        rotate(rig, "Forearm.L", (0, 1, 0), -2.00 * envelope)
        rotate(rig, "Hand.L", (0, 1, 0), 0.3 * math.sin(t * math.tau * 3) * envelope)
        hand = rig.pose.bones["Hand.L"]
        hand.rotation_quaternion = Quaternion((0, 1, 0), math.pi * envelope) @ hand.rotation_quaternion
        rotate(rig, "Head", (0, 1, 0), -0.05 * envelope)
    elif clip == "listen":
        rotate(rig, "Head", (0, 1, 0), -0.115 * envelope)
        rotate(rig, "Arm.R", (1, 0, 0), -0.09 * envelope)
    elif clip == "think":
        rotate(rig, "Head", (0, 0, 1), -0.07 * envelope)
        rotate(rig, "Arm.R", (0, 1, 0), 1.30 * envelope)
        rotate(rig, "Forearm.R", (1, 0, 0), -0.6 * envelope)
        rotate(rig, "Hand.R", (0, 1, 0), 0.5 * envelope)
    elif clip == "speak":
        rotate(rig, "Head", (1, 0, 0), 0.045 * math.sin(t * math.tau * 2) * envelope)
        rotate(rig, "Arm.L", (0, 1, 0), -0.34 * envelope)
        rotate(rig, "Forearm.L", (1, 0, 0), -0.19 * envelope)
        rotate(rig, "Arm.R", (0, 1, 0), 0.14 * envelope)
    elif clip == "encourage":
        rotate(rig, "Head", (1, 0, 0), 0.105 * math.sin(t * math.tau * 2) * envelope)
        rotate(rig, "Arm.L", (0, 1, 0), -0.55 * envelope)
        rotate(rig, "Arm.R", (0, 1, 0), 0.55 * envelope)
    elif clip.startswith("move_"):
        direction = {"move_left": (-1, 0, 0), "move_right": (1, 0, 0), "move_forward": (0, -1, 0), "move_backward": (0, 1, 0)}[clip]
        stride = math.sin(t * math.tau * 4) * envelope
        displacement = Vector(direction) * (0.32 * envelope)
        displacement.z = 0.024 * abs(stride)
        root_bone = rig.pose.bones["Root"]
        root_bone.location = root_bone.bone.matrix_local.to_quaternion().inverted() @ displacement
        rotate(rig, "Leg.L", (1, 0, 0), 0.11 * stride)
        rotate(rig, "Leg.R", (1, 0, 0), -0.11 * stride)
        rotate(rig, "Foot.L", (1, 0, 0), -0.11 * stride)
        rotate(rig, "Foot.R", (1, 0, 0), 0.11 * stride)
        rotate(rig, "Arm.L", (1, 0, 0), -0.15 * stride)
        rotate(rig, "Arm.R", (1, 0, 0), 0.15 * stride)


def animate(rig):
    rig.animation_data_create()
    for name, duration in CLIPS.items():
        action = bpy.data.actions.new(name)
        action.use_fake_user = True
        rig.animation_data.action = action
        last = round(duration * FPS)
        for frame in range(last + 1):
            bpy.context.scene.frame_set(frame)
            pose(rig, name, frame / last)
            for bone in rig.pose.bones:
                bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=bone.name)
                bone.keyframe_insert(data_path="location", frame=frame, group=bone.name)
                bone.keyframe_insert(data_path="scale", frame=frame, group=bone.name)
        track = rig.animation_data.nla_tracks.new()
        track.name = name
        track.strips.new(name, 0, action)
        track.mute = True
    rig.animation_data.action = bpy.data.actions["idle"]
    bpy.context.scene.frame_set(0)


def idea_card(surface, teal):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.98, -0.12, 1.30))
    card = bpy.context.object
    card.name = "IdeaCard"
    card.scale = (0.28, 0.045, 0.22)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = card.modifiers.new("SoftCorners", "BEVEL")
    bevel.width = 0.04
    bevel.segments = 3
    bpy.context.view_layer.objects.active = card
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    card.data.materials.append(surface)
    card["optional"] = True
    card["defaultVisible"] = False
    for index, length in enumerate((0.16, 0.20, 0.12)):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.025, 0.095 - index * 0.09))
        stroke = bpy.context.object
        stroke.name = "IdeaCardLine" + str(index)
        stroke.scale = (length, 0.009, 0.015)
        stroke.parent = card
        stroke.data.materials.append(teal)
    return card


def studio():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 768
    scene.render.resolution_y = 896
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.8, 0.88, 0.85, 1)
    background.inputs["Strength"].default_value = 0.35
    for name, coordinate, power, size in (("Key", (-3, -4, 6), 450, 4), ("Fill", (4, -3, 3.5), 180, 3), ("Rim", (0, 3, 4.5), 300, 3)):
        bpy.ops.object.light_add(type="AREA", location=coordinate)
        light = bpy.context.object
        light.name = name
        light.data.energy = power
        light.data.shape = "DISK"
        light.data.size = size
        light.rotation_euler = (Vector((0, 0, 1.4)) - light.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.camera_add(location=(3.6, -8.2, 3.1))
    camera = bpy.context.object
    camera.name = "ReviewCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.7
    camera.rotation_euler = (Vector((-0.08, 0, 1.5)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    return camera


def export(rig, body, card):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (rig, body, card, *card.children):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=str(EXPORT / "morii.glb"),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_nla_strips=True,
        export_def_bones=True,
        export_skins=True,
        export_influence_nb=4,
        export_morph=True,
        export_morph_animation=False,
        export_morph_normal=False,
        export_texcoords=False,
        export_tangents=False,
        export_extras=True,
        export_cameras=False,
        export_lights=False,
        export_materials="EXPORT",
        export_force_sampling=True,
        export_frame_step=1,
        export_optimize_animation_size=True,
        export_armature_object_remove=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-render", action="store_true")
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    for directory in (EXPORT, PREVIEW, REVIEW):
        directory.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.fps = FPS
    surfaces = [
        material("MoriiTeal", (0.14, 0.40, 0.39), 0.78),
        material("WarmCream", (0.86, 0.73, 0.59), 0.79),
        material("SoftPeach", (0.94, 0.43, 0.27), 0.76),
        material("Cocoa", (0.055, 0.027, 0.02), 0.4),
        material("EyeCream", (0.93, 0.91, 0.82), 0.24),
        material("WarmEyes", (0.028, 0.013, 0.010), 0.28),
        material("IdeaPaper", (0.9, 0.87, 0.75), 0.88),
    ]
    body = make_character(surfaces)
    rig = make_rig(body)
    animate(rig)
    card = idea_card(surfaces[-1], surfaces[0])
    camera = studio()
    export(rig, body, card)
    card.hide_render = True
    for child in card.children:
        child.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "source" / "morii" / "morii.blend"))
    glb = (EXPORT / "morii.glb").read_bytes()
    document = json.loads(glb[20:20 + struct.unpack_from("<I", glb, 12)[0]])
    triangles = sum(document["accessors"][primitive["indices"]]["count"] // 3 for mesh in document["meshes"] for primitive in mesh["primitives"])
    minimum = [min(vertex.co[axis] for vertex in body.data.vertices) for axis in range(3)]
    maximum = [max(vertex.co[axis] for vertex in body.data.vertices) for axis in range(3)]
    metadata = {
        "asset": "morii.glb",
        "generator": "Blender " + bpy.app.version_string,
        "triangles": triangles,
        "bytes": (EXPORT / "morii.glb").stat().st_size,
        "clips": {name: round(seconds * FPS) / FPS for name, seconds in CLIPS.items()},
        "bones": list(rig.data.bones.keys()),
        "morphs": ["blink", "mouthOpen"],
        "optionalNodes": ["IdeaCard"],
        "maxTravelMeters": 0.32,
        "restHeightMeters": maximum[2] - minimum[2],
        "boundsYUp": {
            "min": [minimum[0], minimum[2], -maximum[1]],
            "max": [maximum[0], maximum[2], -minimum[1]],
        },
        "notes": "Runtime must hide IdeaCard by default. Voice mouth movement is a cue, not phoneme lip sync.",
    }
    (EXPORT / "morii.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not arguments.skip_render:
        frames = (("idle", 0, "front"), ("wave", 28, "wave"), ("wave", 28, "wave-front"), ("listen", 28, "listen"), ("think", 34, "think"), ("speak", 28, "speak"), ("move_left", 36, "move-left"))
        for clip, frame, name in frames:
            rig.animation_data.action = bpy.data.actions[clip]
            bpy.context.scene.frame_set(frame)
            body.data.shape_keys.key_blocks["mouthOpen"].value = 0.55 if clip == "speak" else 0
            if name in ("front", "wave-front"):
                camera.location = (0, -8, 2.65)
            else:
                camera.location = (3.6, -8.2, 3.1)
            camera.rotation_euler = (Vector((-0.08, 0, 1.5)) - camera.location).to_track_quat("-Z", "Y").to_euler()
            bpy.context.scene.render.filepath = str(REVIEW / (name + ".png"))
            bpy.ops.render.render(write_still=True)
            if name == "front":
                (PREVIEW / "morii-3d.png").write_bytes((REVIEW / "front.png").read_bytes())
    print("MORII_BUILD " + json.dumps(metadata))


if __name__ == "__main__":
    main()
