"""Read-only structural and deformation audit of the existing Morii source."""
import json
from collections import Counter
from pathlib import Path
import bpy

output = Path(__file__).parent / 'existing-blend-structure.json'
body = bpy.data.objects['MoriiBody']
rig = bpy.data.objects['MoriiRig']
mesh = body.data
parent = list(range(len(mesh.vertices)))
def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for edge in mesh.edges:
    a, b = map(root, edge.vertices)
    parent[a] = b
components = {}
for v in mesh.vertices:
    components.setdefault(root(v.index), []).append(v.index)
edge_faces = Counter()
for poly in mesh.polygons:
    for edge in poly.edge_keys:
        edge_faces[tuple(sorted(edge))] += 1
islands = []
for indices in components.values():
    groups = Counter()
    for i in indices:
        for g in mesh.vertices[i].groups:
            if g.weight > .01:
                groups[body.vertex_groups[g.group].name] += 1
    islands.append({'vertices': len(indices), 'groups': dict(groups), 'min': [min(mesh.vertices[i].co[a] for i in indices) for a in range(3)], 'max': [max(mesh.vertices[i].co[a] for i in indices) for a in range(3)]})
islands.sort(key=lambda item: item['vertices'], reverse=True)
material_data = []
for mat in bpy.data.materials:
    shader = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if mat.use_nodes else None
    material_data.append({'name': mat.name, 'baseColor': list(shader.inputs['Base Color'].default_value) if shader else list(mat.diffuse_color), 'roughness': shader.inputs['Roughness'].default_value if shader else None})
def coordinates():
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    temp = evaluated.to_mesh()
    values = [v.co.copy() for v in temp.vertices]
    evaluated.to_mesh_clear()
    return values
rig.animation_data.action = bpy.data.actions['idle']
bpy.context.scene.frame_set(0)
neutral = coordinates()
edge_lengths = [(neutral[e.vertices[0]] - neutral[e.vertices[1]]).length for e in mesh.edges]
deformation = []
for name in ['wave', 'think', 'speak', 'move_forward']:
    action = bpy.data.actions[name]
    rig.animation_data.action = action
    bpy.context.scene.frame_set(round((action.frame_range[0] + action.frame_range[1]) * .5))
    values = coordinates()
    ratios = []
    per_group = {}
    for edge, length in zip(mesh.edges, edge_lengths):
        if length < 1e-6:
            continue
        ratio = (values[edge.vertices[0]] - values[edge.vertices[1]]).length / length
        ratios.append(ratio)
        for i in edge.vertices:
            influences = mesh.vertices[i].groups
            if influences:
                g = max(influences, key=lambda g: g.weight)
                group_name = body.vertex_groups[g.group].name
                per_group.setdefault(group_name, []).append(ratio)
    ratios.sort()
    deformation.append({'action': name, 'minEdgeLengthRatio': ratios[0], 'maxEdgeLengthRatio': ratios[-1], 'fractionEdgesOutside0_8to1_2': sum(r < .8 or r > 1.2 for r in ratios)/len(ratios), 'groupExtrema': {name: {'min': min(rs), 'max': max(rs)} for name, rs in per_group.items() if max(rs) > 1.2 or min(rs) < .8}})
shape_keys = []
basis = mesh.shape_keys.key_blocks['Basis']
for key in mesh.shape_keys.key_blocks:
    changed = [i for i, v in enumerate(key.data) if (v.co - basis.data[i].co).length > 1e-7]
    shape_keys.append({'name': key.name, 'changedVertices': len(changed), 'maxOffset': max(((key.data[i].co - basis.data[i].co).length for i in changed), default=0)})
camera = bpy.context.scene.camera
report = {'source': bpy.data.filepath, 'version': bpy.app.version_string, 'mesh': {'vertices': len(mesh.vertices), 'edges': len(mesh.edges), 'faces': len(mesh.polygons), 'faceArities': dict(Counter(len(p.vertices) for p in mesh.polygons)), 'connectedComponents': len(components), 'boundaryEdges': sum(n == 1 for n in edge_faces.values()), 'nonManifoldEdgesAbove2Faces': sum(n > 2 for n in edge_faces.values()), 'materialSlots': [s.material.name for s in body.material_slots], 'islands': islands}, 'materials': material_data, 'shapeKeys': shape_keys, 'deformation': deformation, 'camera': {'name': camera.name, 'type': camera.data.type, 'orthoScale': camera.data.ortho_scale, 'location': list(camera.location), 'rotationRadians': list(camera.rotation_euler)}, 'render': {'engine': bpy.context.scene.render.engine, 'viewTransform': bpy.context.scene.view_settings.view_transform, 'exposure': bpy.context.scene.view_settings.exposure, 'resolution': [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y]}, 'limits': ['No source or live Blender scene was saved or modified.', 'Edge length changes flag possible deformation issues, but do not prove visibly unacceptable joints.', 'Disconnected components are allowed for accessories/markings; they do not constitute errors by themselves.', 'Reference PNGs do not uniquely determine hidden 3D surfaces.']}
output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('AUDIT_SAVED ' + str(output))
