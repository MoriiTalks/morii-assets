"""Check neutral clip boundaries, small travel bounds, and grounded geometry."""

import json

import bpy
from mathutils import Vector


rig = bpy.data.objects["MoriiRig"]
body = bpy.data.objects["MoriiBody"]
assert body.data.shape_keys.key_blocks["blink"].value == 0
assert body.data.shape_keys.key_blocks["mouthOpen"].value == 0
expected = {
    "idle", "wave", "listen", "think", "speak", "encourage",
    "move_left", "move_right", "move_forward", "move_backward", "reset",
}
assert set(bpy.data.actions.keys()) == expected
summary = []
minimum_ground = 1
for action in bpy.data.actions:
    rig.animation_data.action = action
    first, last = (round(value) for value in action.frame_range)
    for frame in (first, last):
        bpy.context.scene.frame_set(frame)
        for bone in rig.pose.bones:
            assert bone.location.length < 1e-5, (action.name, frame, bone.name, "translation")
            assert abs(bone.rotation_quaternion.w - 1) < 1e-5, (action.name, frame, bone.name, "rotation")
            assert (bone.scale - Vector((1, 1, 1))).length < 1e-5, (action.name, frame, bone.name, "scale")
    maximum = 0
    lowest = 1
    root_minimum = [0, 0, 0]
    root_maximum = [0, 0, 0]
    frames = range(first, last + 1) if action.name.startswith("move_") else (first, round((first + last) / 2), last)
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        root = rig.pose.bones["Root"]
        displacement = root.bone.matrix_local.to_quaternion() @ root.location
        yup = (displacement.x, displacement.z, -displacement.y)
        root_minimum = [min(a, b) for a, b in zip(root_minimum, yup)]
        root_maximum = [max(a, b) for a, b in zip(root_maximum, yup)]
        maximum = max(maximum, abs(displacement.x), abs(displacement.y))
        evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        lowest = min(lowest, min(vertex.co.z for vertex in evaluated.data.vertices))
        if action.name == "move_forward":
            assert -0.321 < displacement.y <= 1e-5
        if action.name == "move_backward":
            assert -1e-5 <= displacement.y < 0.321
        if action.name == "move_left":
            assert -0.321 < displacement.x <= 1e-5
        if action.name == "move_right":
            assert -1e-5 <= displacement.x < 0.321
    assert maximum <= 0.321, (action.name, maximum)
    assert lowest >= -0.012, (action.name, lowest)
    minimum_ground = min(minimum_ground, lowest)
    summary.append({"clip": action.name, "seconds": (last - first) / bpy.context.scene.render.fps, "travel": maximum, "rootBoundsYUp": {"min": root_minimum, "max": root_maximum}, "lowestVertex": lowest})
print("MORII_RIG_VERIFIED " + json.dumps({"clips": summary, "minimumGround": minimum_ground}))
