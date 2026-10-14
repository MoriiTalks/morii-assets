"""Reference-driven Morii geometry study, execution belongs to the live Blender MCP scene."""

import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'validation' / 'likeness-2026-10-14'
SOURCE = ROOT / 'source' / 'morii' / 'reconstruction'
PPM = 350.0
HEAD_ORIGIN = Vector((644, 364))
ROLL = math.radians(10.8)
PARTS = []
MARKERS = {}
HEAD_SURFACE = {}


def pixel(point, depth=0):
    return Vector(((point[0] - 640) / PPM, depth, (1164 - point[1]) / PPM))


def srgb(value):
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def material(name, color, roughness=0.65):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = next(node for node in result.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    rgb = tuple(srgb(channel / 255) for channel in color)
    shader.inputs['Base Color'].default_value = (*rgb, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Specular IOR Level'].default_value = 0.3
    result.diffuse_color = (*rgb, 1)
    return result


def plush_surface(surface):
    # Preview micro-normal detail is cosmetic, reconstruction boundaries remain explicit geometry.
    tree=surface.node_tree
    shader=next(node for node in tree.nodes if node.type=='BSDF_PRINCIPLED')
    texture=tree.nodes.new('ShaderNodeTexNoise')
    texture.inputs['Scale'].default_value=185
    texture.inputs['Detail'].default_value=2
    bump=tree.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value=.16
    bump.inputs['Distance'].default_value=.018
    tree.links.new(texture.outputs['Fac'],bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'],shader.inputs['Normal'])
    surface['preview_only_detail']='Procedural micro-normal needs baking before approved runtime export'


def mesh(name, vertices, faces, materials, bone, indices=None):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    for surface in materials:
        data.materials.append(surface)
    for polygon in data.polygons:
        polygon.use_smooth = True
        if indices is not None:
            polygon.material_index = indices[polygon.index]
    if bone:
        group = obj.vertex_groups.new(name=bone)
        group.add(list(range(len(vertices))), 1, 'REPLACE')
    PARTS.append(obj)
    return obj


def catmull(points, t, closed=False):
    length = len(points) if closed else len(points) - 1
    scaled = min(t, 0.999999) * length
    index, amount = int(scaled), scaled % 1
    def value(offset):
        position = (index + offset) % len(points) if closed else max(0, min(len(points) - 1, index + offset))
        return Vector(points[position])
    a, b, c, d = (value(offset) for offset in (-1, 0, 1, 2))
    return 0.5 * ((2 * b) + (-a + c) * amount + (2 * a - 5 * b + 4 * c - d) * amount ** 2 + (-a + 3 * b - 3 * c + d) * amount ** 3)


def unroll(point):
    offset = Vector(point) - HEAD_ORIGIN
    return Vector((offset.x * math.cos(ROLL) - offset.y * math.sin(ROLL), offset.x * math.sin(ROLL) + offset.y * math.cos(ROLL))) / PPM


def head_position(x, z, depth):
    return Vector((x, depth, z)) + pixel(HEAD_ORIGIN)


def head_depth(x, z, radius=None):
    if radius is None and 'tree' in HEAD_SURFACE:
        location, _normal, _face, _distance = HEAD_SURFACE['tree'].ray_cast(Vector((x, -3, z)) + pixel(HEAD_ORIGIN), Vector((0, 1, 0)))
        if location is not None:
            return location.y
    if radius is None:
        radius = min(0.999, math.sqrt((x / 0.84) ** 2 + ((z - 0.025) / 0.67) ** 2))
    roundness = math.sqrt(max(0, 1 - radius ** 2))
    cheek = 0.20 * math.exp(-((z + 0.27) / 0.26) ** 2) * math.exp(-(x / 0.75) ** 6) * roundness
    return -0.12 - 0.47 * roundness - cheek


def cream_boundary(x):
    points=sorted((unroll(point).x,-unroll(point).y) for point in [(360,485),(385,441),(440,419),(490,421),(530,425),(603,373),(662,358),(716,361),(770,375),(831,344),(868,340),(900,353),(923,384)])
    if x<=points[0][0]:return points[0][1]
    if x>=points[-1][0]:return points[-1][1]
    for first,last in zip(points,points[1:]):
        if first[0]<=x<=last[0]:
            amount=(x-first[0])/(last[0]-first[0])
            index=points.index(first)
            previous=points[max(0,index-1)]
            following=points[min(len(points)-1,index+2)]
            left=(last[1]-previous[1])/(last[0]-previous[0])
            right=(following[1]-first[1])/(following[0]-first[0])
            span=last[0]-first[0]
            return (2*amount**3-3*amount**2+1)*first[1]+(amount**3-2*amount**2+amount)*span*left+(-2*amount**3+3*amount**2)*last[1]+(amount**3-amount**2)*span*right
    return points[-1][1]


def blend(a, b, amount):
    amount = min(1, max(0, amount))
    amount = amount * amount * (3 - 2 * amount)
    return tuple(first + (second-first) * amount for first, second in zip(a, b))


def vertex_palette(obj, colors, surfaces):
    # The marking belongs to the surface, vertex colors remove intersecting pigment shells.
    attribute = obj.data.color_attributes.new(name='MoriiPalette', type='FLOAT_COLOR', domain='POINT')
    for value, color in zip(attribute.data, colors):
        value.color = (*color, 1)
    obj.data.materials.clear()
    surface = surfaces[0].copy()
    surface.name = obj.name + 'Palette'
    tree = surface.node_tree
    shader = next(node for node in tree.nodes if node.type == 'BSDF_PRINCIPLED')
    node = tree.nodes.new('ShaderNodeVertexColor')
    node.layer_name = 'MoriiPalette'
    tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
    obj.data.materials.append(surface)
    for polygon in obj.data.polygons:polygon.material_index=0


def pigment_texture(obj, surfaces, x_range, z_range, origin, face=False):
    # UV pigment controls color alone, silhouette, muzzle depth and facial features remain geometry.
    width,height=1024,512
    image=bpy.data.images.new(obj.name+'Pigment',width=width,height=height,alpha=False)
    image.colorspace_settings.name='Non-Color'
    rgb=[tuple(channel for channel in surface.diffuse_color[:3]) for surface in surfaces[:3]]
    colors=rgb
    pixels=[]
    for row in range(height):
        z=z_range[0]+(z_range[1]-z_range[0])*row/(height-1)
        for column in range(width):
            front=column<width//2
            x=x_range[0]+(x_range[1]-x_range[0])*(column%(width//2))/(width//2-1)
            if face:
                field=(cream_boundary(x)-z)/.006+.5
                centers=[unroll(point) for point in [(452,458),(854,382)]]
                blush=max(1-((x-point.x)/.099)**2-((z+point.y)/.072)**2 for point in centers)
            else:
                center=.11+.05*(1-min(1,max(0,(z-.4)/1.1)))
                field=(1-((x-center)/(.49*(1-.22*max(0,z-1.00))))**2-((z-1.15)/.78)**2)/.014+.5
                blush=-1
            color=blend(colors[0],colors[1],field if front else 0)
            if face and front:color=blend(color,colors[2],blush/.045)
            pixels.extend((*color,1))
    image.pixels.foreach_set(pixels)
    image.pack()
    layer=obj.data.uv_layers.new(name='PigmentUV')
    for polygon in obj.data.polygons:
        center=polygon.center-Vector(origin)
        front=center.y<-.12 if face else center.y<0
        for index in polygon.loop_indices:
            point=obj.data.vertices[obj.data.loops[index].vertex_index].co-Vector(origin)
            u=(point.x-x_range[0])/(x_range[1]-x_range[0])*.5+(0 if front else .5)
            v=(point.z-z_range[0])/(z_range[1]-z_range[0])
            layer.data[index].uv=(u,v)
    surface=surfaces[0].copy();surface.name=obj.name+'PigmentMaterial'
    tree=surface.node_tree
    shader=next(node for node in tree.nodes if node.type=='BSDF_PRINCIPLED')
    texture=tree.nodes.new('ShaderNodeTexImage');texture.image=image
    tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
    obj.data.materials.clear();obj.data.materials.append(surface)
    for polygon in obj.data.polygons:polygon.material_index=0


def head(surfaces):
    outline = [(632,138),(744,155),(825,209),(883,277),(929,377),(945,456),(918,508),(872,551),(788,582),(672,600),(557,595),(469,574),(412,539),(372,498),(348,436),(349,373),(370,303),(414,228),(498,170),(574,143)]
    contour = [unroll(point) for point in outline]
    radial, rings = 128, 36
    vertices, faces, colors = [], [], []
    center = Vector((-0.025, 0.035))
    for back in (False, True):
        offset = len(vertices)
        vertices.append(head_position(center.x, -center.y, 0.39 if back else head_depth(center.x, -center.y, 0)))
        for ring in range(1, rings + 1):
            radius = ring / rings
            for segment in range(radial):
                edge = catmull(contour, segment / radial, True)
                x, down = center + (edge - center) * radius
                z = -down
                depth = -0.12 + 0.51 * math.sqrt(max(0, 1 - radius ** 2)) if back else head_depth(x, z, radius)
                vertices.append(head_position(x, z, depth))
        for ring in range(1, rings + 1):
            for segment in range(radial):
                current = offset + 1 + (ring - 1) * radial + segment
                following = offset + 1 + (ring - 1) * radial + (segment + 1) % radial
                face = (offset, current, following) if ring == 1 else (current - radial, current, following, following - radial)
                if not back:
                    face = tuple(reversed(face))
                faces.append(face)
                midpoint = sum((vertices[index] for index in face), Vector()) / len(face) - pixel(HEAD_ORIGIN)
                blush = any(((midpoint.x - sign * 0.57) / 0.099) ** 2 + ((midpoint.z + 0.245) / 0.072) ** 2 < 1 for sign in (-1, 1))
                colors.append(2 if blush and not back else 1 if midpoint.z < cream_boundary(midpoint.x) and not back else 0)
    side_offset = 1 + rings * radial
    for segment in range(radial):
        a = 1 + (rings - 1) * radial + segment
        b = 1 + (rings - 1) * radial + (segment + 1) % radial
        faces.append((a, b, b + side_offset, a + side_offset))
        colors.append(0)
    obj = mesh('MoriiHeadSurface', vertices, faces, surfaces[:3], 'Head', colors)
    palette=[tuple(next(node for node in surface.node_tree.nodes if node.type=='BSDF_PRINCIPLED').inputs['Base Color'].default_value[:3]) for surface in surfaces[:3]]
    vertex_colors=[]
    for vertex in obj.data.vertices:
        point=vertex.co-pixel(HEAD_ORIGIN)
        front=point.y < -.121
        amount=(cream_boundary(point.x)-point.z)/.018+.5 if front else 0
        color=blend(palette[0],palette[1],amount)
        blush=max(1-((point.x-sign*.57)/.099)**2-((point.z+.245)/.072)**2 for sign in (-1,1)) if front else -1
        vertex_colors.append(blend(color,palette[2],blush/.10))
    pigment_texture(obj,surfaces,(-.9,.9),(-.75,.75),pixel(HEAD_ORIGIN),True)
    HEAD_SURFACE['tree']=BVHTree.FromObject(obj,bpy.context.evaluated_depsgraph_get())
    obj['construction'] = 'Closed front/back quad surface from measured wave head contour, no separate muzzle shell'
    return obj


def surface_feature(name, center, scale, surface, bone, rotation=None, segments=40, rings=24):
    vertices, faces = [], []
    center = Vector(center)
    orientation = rotation or Quaternion()
    for ring in range(rings + 1):
        latitude = math.pi * ring / rings
        for segment in range(segments):
            longitude = math.tau * segment / segments
            point = Vector((scale[0] * math.sin(latitude) * math.cos(longitude), scale[1] * math.sin(latitude) * math.sin(longitude), scale[2] * math.cos(latitude)))
            vertices.append(center + orientation @ point)
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    return mesh(name, vertices, faces, [surface], bone)


def face_point(point, front=0):
    x, down = unroll(point)
    return head_position(x, -down, head_depth(x, -down) - front)


def curve(name, points, thickness, surface, bone, radii=None):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 16
    data.bevel_depth = thickness
    data.bevel_resolution = 4
    spline = data.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for index, (point, coordinate) in enumerate(zip(spline.bezier_points, points)):
        point.co = coordinate
        point.handle_left_type = 'AUTO'
        point.handle_right_type = 'AUTO'
        if radii:
            point.radius = radii[index]
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.object
    obj.data.materials.append(surface)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    group = obj.vertex_groups.new(name=bone)
    group.add(list(range(len(obj.data.vertices))), 1, 'REPLACE')
    PARTS.append(obj)
    return obj


def facial_features(surfaces):
    teal, cream, pink, brown, eyes, highlight = surfaces
    for suffix, point in [('R', (494, 393)), ('L', (795, 337))]:
        center = face_point(point, 0.014)
        x, down = unroll(point)
        surface_feature('EyeWhite.' + suffix, center, (0.105, 0.046, 0.114), cream, 'Head')
        pupil = surface_feature('Eye.' + suffix, center + Vector((0.004, -0.046, 0.008)), (0.095, 0.039, 0.106), eyes, 'Head')
        pupil['feature'] = 'eye'
        surface_feature('Highlight.' + suffix, center + Vector((0.026, -0.080, 0.042)), (0.022, 0.006, 0.022), highlight, 'Head', segments=24, rings=16)
        MARKERS['eye_' + ('left' if suffix == 'R' else 'right')] = ('Head', center)
    nose_center = face_point((652, 390), 0.047)
    nose = surface_feature('Nose', nose_center, (0.153, 0.078, 0.087), brown, 'Head', segments=48, rings=28)
    for vertex in nose.data.vertices:
        local = vertex.co - nose_center
        local.x *= 1 + 0.18 * local.z / 0.087
        vertex.co = nose_center + local
    MARKERS['nose'] = ('Head', nose_center)
    curve('Philtrum', [face_point(point, 0.015) for point in [(657,418),(660,443),(661,452)]], 0.008, brown, 'Head')
    for suffix, points in [('R',[(660,450),(625,470),(586,472),(558,457)]),('L',[(660,450),(689,458),(728,447),(754,417)])]:
        curve('Smile.' + suffix, [face_point(point, 0.014) for point in points], 0.007, brown, 'Head', [0.9,1,1,0.6])
    for suffix, points in [('R',[(454,317),(463,296),(487,291)]),('L',[(761,239),(782,236),(801,248)])]:
        curve('Brow.' + suffix, [face_point(point, 0.006) for point in points], 0.016, brown, 'Head', [0.25,1,0.55])
    whiskers = [([(405,473),(376,483),(348,503)], 'R'), ([(410,496),(378,514),(363,538)],'R'), ([(414,515),(392,540),(383,561)],'R'), ([(901,380),(926,374),(947,379)],'L'), ([(905,402),(931,407),(947,422)],'L'), ([(905,425),(922,445),(938,465)],'L')]
    for index, (points, suffix) in enumerate(whiskers):
        coords = [face_point(points[0], 0.008)]
        for point in points[1:]:
            x, down = unroll(point)
            coords.append(head_position(x, -down, coords[0].y + 0.012))
        curve('Whisker' + str(index), coords, 0.0055, brown, 'Head', [1,0.85,0.25])


def ear(suffix, center_pixel, surfaces):
    center = face_point(center_pixel, -0.18)
    teal, cream = surfaces[:2]
    radial, rings = 64, 16
    vertices, faces, indices = [], [], []
    for back in (False, True):
        offset = len(vertices)
        vertices.append(center + Vector((0, 0.08 if not back else 0.14, 0)))
        for ring in range(1, rings + 1):
            radius = ring / rings
            depth = .08*(1-radius)**2+.035*radius if not back else 0.14 + 0.025 * math.sqrt(1-radius**2)
            for segment in range(radial):
                angle = segment * math.tau / radial
                vertices.append(center + Vector((0.17 * radius * math.cos(angle), depth, 0.195 * radius * math.sin(angle))))
        for ring in range(1, rings + 1):
            for segment in range(radial):
                a = offset + 1 + (ring - 1) * radial + segment
                b = offset + 1 + (ring - 1) * radial + (segment + 1) % radial
                face = (offset,a,b) if ring==1 else (a-radial,a,b,b-radial)
                faces.append(face if not back else tuple(reversed(face)))
                indices.append(1 if not back and ring <= 10 else 0)
    offset=1+rings*radial
    for segment in range(radial):
        a=1+(rings-1)*radial+segment
        b=1+(rings-1)*radial+(segment+1)%radial
        faces.append((a,b,b+offset,a+offset))
        indices.append(0)
    return mesh('Ear.'+suffix,vertices,faces,[teal,cream],'Head',indices)


def tube(name, points, radii, surface, bone, flat=1, segments=64, radial=32, round_end=False):
    vertices, faces = [], []
    for ring in range(segments+1):
        t=ring/segments
        center=catmull(points,t)
        tangent=(catmull(points,min(1,t+0.001))-catmull(points,max(0,t-0.001))).normalized()
        normal=tangent.cross(Vector((0,1,0))).normalized()
        if normal.length<0.1: normal=tangent.cross(Vector((1,0,0))).normalized()
        depth=normal.cross(tangent).normalized()
        position=t*(len(radii)-1)
        index=min(len(radii)-2,int(position))
        radius=radii[index]+(radii[index+1]-radii[index])*(position-index)
        if round_end and t>=(len(radii)-2)/(len(radii)-1):
            amount=t*(len(radii)-1)-(len(radii)-2)
            radius=max(.001,radii[-2]*math.sqrt(max(0,1-amount**2)))
        for segment in range(radial):
            angle=math.tau*segment/radial
            vertices.append(center+radius*(normal*math.cos(angle)+depth*math.sin(angle)*flat))
        if ring:
            for segment in range(radial):
                a=(ring-1)*radial+segment
                b=(ring-1)*radial+(segment+1)%radial
                faces.append((a,b,b+radial,a+radial))
    faces.extend([tuple(reversed(range(radial))),tuple(range(segments*radial,(segments+1)*radial))])
    return mesh(name,vertices,faces,[surface],bone)


def torso(surfaces):
    sections=[(0.40,0.36,0.24,0.19),(0.52,0.53,0.30,0.18),(0.72,0.63,0.34,0.17),(0.95,0.60,0.35,0.14),(1.18,0.50,0.30,0.09),(1.45,0.38,0.25,0.03),(1.72,0.31,0.24,0),(1.87,0.28,0.23,0)]
    vertices,faces,colors=[],[],[]
    radial,rings=96,72
    for ring in range(rings+1):
        z,rx,ry,cx=catmull(sections,ring/rings)
        for segment in range(radial):
            angle=math.tau*segment/radial
            x=cx+rx*math.cos(angle)
            y=0.04+ry*math.sin(angle)
            vertices.append((x,y,z))
        if ring:
            for segment in range(radial):
                a=(ring-1)*radial+segment
                b=(ring-1)*radial+(segment+1)%radial
                faces.append((a,b,b+radial,a+radial))
                point=sum((Vector(vertices[index]) for index in faces[-1]),Vector())/4
                marking=((point.x-0.08)/(0.49*(1-0.22*max(0,point.z-1.00))))**2+((point.z-1.15)/0.78)**2<1 and point.y<0
                colors.append(1 if marking else 0)
    faces.extend([tuple(reversed(range(radial))),tuple(range(rings*radial,(rings+1)*radial))])
    colors.extend([0,0])
    obj=mesh('Torso',vertices,faces,surfaces[:2],'Spine',colors)
    palette=[tuple(next(node for node in surface.node_tree.nodes if node.type=='BSDF_PRINCIPLED').inputs['Base Color'].default_value[:3]) for surface in surfaces[:2]]
    vertex_colors=[]
    for vertex in obj.data.vertices:
        point=vertex.co
        field=1-((point.x-.08)/(.49*(1-.22*max(0,point.z-1.00))))**2-((point.z-1.15)/.78)**2
        vertex_colors.append(blend(palette[0],palette[1],field/.05+.5 if point.y<0 else 0))
    pigment_texture(obj,surfaces,(-.9,.9),(0,2.1),(0,0,0))
    return obj


def tail(surfaces):
    # The buried root joins the hip, the underside pigment follows the curved surface UVs.
    obj=tube('TailSurface',[(.05,.34,.69),(-.43,.30,.49),(-.89,.24,.62),(-1.16,.17,.90)],[.045,.235,.165,.002],surfaces[0],'Tail',flat=.82,segments=96,radial=48,round_end=True)
    tip_group=obj.vertex_groups.new(name='TailTip')
    for vertex in obj.data.vertices:
        amount=max(0,min(1,(-vertex.co.x-.4)/.65))
        obj.vertex_groups['Tail'].add([vertex.index],1-amount,'REPLACE')
        tip_group.add([vertex.index],amount,'REPLACE')
    image=bpy.data.images.new('TailUndersidePigment',width=512,height=256,alpha=False)
    image.colorspace_settings.name='Non-Color'
    colors=[]
    for material in surfaces[:2]:
        colors.append(tuple(material.diffuse_color[:3]))
    pixels=[]
    for row in range(256):
        angle=math.tau*row/255
        for column in range(512):
            amount=column/511
            cream=(math.cos(angle)-.40)/.025+.5
            cream*=min(1,max(0,(amount-.12)/.08))
            color=blend(colors[0],colors[1],cream)
            pixels.extend((*color,1))
    image.pixels.foreach_set(pixels);image.pack()
    uv=obj.data.uv_layers.new(name='TailSurfaceUV')
    for polygon in obj.data.polygons:
        seam=any(obj.data.loops[index].vertex_index%48==47 for index in polygon.loop_indices)
        for index in polygon.loop_indices:
            vertex=obj.data.loops[index].vertex_index
            angle=vertex%48
            uv.data[index].uv=(vertex//48/96,(48 if seam and angle==0 else angle)/48)
    surface=surfaces[0].copy();surface.name='TailContinuousPigment'
    shader=next(node for node in surface.node_tree.nodes if node.type=='BSDF_PRINCIPLED')
    texture=surface.node_tree.nodes.new('ShaderNodeTexImage');texture.image=image
    surface.node_tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
    obj.data.materials.clear();obj.data.materials.append(surface)
    return obj


def tufts(surfaces):
    # These two swept lobes use the measured hair silhouette, with their roots inside the head.
    for index,points in enumerate([[(550,162),(581,127),(600,104)],[(585,155),(620,132),(645,120)]]):
        coordinates=[]
        for point in points:
            x,down=unroll(point)
            coordinates.append(head_position(x,-down,-.015))
        tube('Tuft'+str(index),coordinates,[.072,.060,.004],surfaces[0],'Head',flat=.64,segments=40,radial=32)


def sculpt_feet(obj):
    # Shared mesh displacement rounds the toes after body fusion, without separate toe spheres.
    for vertex in obj.data.vertices:
        point=vertex.co
        if point.z<.24 and point.y<-.24:
            center=.075+(.43 if point.x>.075 else -.43)
            local=point.x-center
            front=min(1,max(0,(-point.y-.24)/.08))
            height=min(1,max(0,(.24-point.z)/.06))
            grooves=sum(math.exp(-((local-offset)/.014)**2) for offset in (-.075,.075))
            point.y+=.018*grooves*front*height
    obj.data.update()


def fuse_body(surfaces):
    pieces=[obj for obj in PARTS if obj.name=='Torso' or obj.name.startswith(('LegSurface.','FootSurface.'))]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in pieces:obj.select_set(True);PARTS.remove(obj)
    bpy.context.view_layer.objects.active=pieces[0]
    bpy.ops.object.join()
    obj=bpy.context.object;obj.name='MoriiContinuousBody'
    modifier=obj.modifiers.new('ContinuousAnatomy','REMESH');modifier.mode='VOXEL';modifier.voxel_size=.018;modifier.use_smooth_shade=True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    modifier=obj.modifiers.new('RoundedTransitions','SMOOTH');modifier.factor=.55;modifier.iterations=6
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    modifier=obj.modifiers.new('ReviewBudget','DECIMATE');modifier.ratio=.55
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    sculpt_feet(obj)
    obj.vertex_groups.clear()
    groups={name:obj.vertex_groups.new(name=name) for name in ['Spine','Leg.L','Foot.L','Leg.R','Foot.R']}
    for vertex in obj.data.vertices:
        point=vertex.co
        suffix='L' if point.x>.075 else 'R'
        if point.z>.58:groups['Spine'].add([vertex.index],1,'REPLACE')
        else:
            amount=min(1,max(0,(point.z-.16)/.30))
            groups['Leg.'+suffix].add([vertex.index],amount,'REPLACE')
            groups['Foot.'+suffix].add([vertex.index],1-amount,'REPLACE')
    for polygon in obj.data.polygons:polygon.use_smooth=True
    modifier=obj.modifiers.new('ReviewSurface','SUBSURF');modifier.levels=1;modifier.render_levels=1
    pigment_texture(obj,surfaces,(-.9,.9),(0,2.1),(0,0,0))
    PARTS.append(obj)


def paw(name, center, direction, normal, surface, bone, foot=False):
    axis=Vector(direction).normalized()
    normal=(Vector(normal)-axis*Vector(normal).dot(axis)).normalized()
    side=normal.cross(axis).normalized()
    rotation=Matrix((side,normal,axis)).transposed().to_quaternion()
    obj=surface_feature(name,center,(0.225 if foot else 0.205,0.145 if foot else .125,0.16 if foot else 0.23),surface,bone,rotation,segments=64,rings=36)
    # Shallow sculpted grooves preserve the broad continuous paw instead of separate toe balls.
    origin=Vector(center)
    inverse=rotation.inverted()
    for vertex in obj.data.vertices:
        point=inverse@(vertex.co-origin)
        if point.z>0.04 and not foot:
            groove=sum(math.exp(-((point.x-offset)/0.015)**2) for offset in (-0.072,0.072))
            point.y*=1-0.22*groove*min(1,(point.z-0.04)/0.08)
            point.z-=.025*groove*min(1,(point.z-.04)/.10)
        vertex.co=origin+rotation@point
    return obj,axis,normal,side,rotation


def arm(suffix,surfaces):
    sign=1 if suffix=='L' else -1
    shoulder=Vector((.08+sign*0.36,0.015,1.49))
    elbow=Vector((.08+sign*0.53,-0.035,1.18))
    wrist=Vector((.08+sign*0.56,-0.06,.86))
    direction=Vector((sign*0.36,0,-0.94)).normalized()
    hand=wrist+direction*0.145
    root=Vector((.08+sign*.29,.04,1.52))
    obj=tube('ArmSurface.'+suffix,[root,shoulder,elbow,wrist,hand,hand+direction*.23],[.025,.18,.20,.145,.205,.001],surfaces[0],'Arm.'+suffix,flat=.74,segments=128,radial=48)
    groups={name:obj.vertex_groups.new(name=name) if name!='Arm.'+suffix else obj.vertex_groups[name] for name in ['Arm.'+suffix,'Forearm.'+suffix,'Hand.'+suffix]}
    for vertex in obj.data.vertices:
        amount=min(1,max(0,(1.24-vertex.co.z)/.12))
        amount=amount*amount*(3-2*amount)
        hand_weight=min(1,max(0,(.94-vertex.co.z)/.11))
        hand_weight=hand_weight*hand_weight*(3-2*hand_weight)
        groups['Arm.'+suffix].add([vertex.index],1-amount,'REPLACE')
        groups['Forearm.'+suffix].add([vertex.index],amount*(1-hand_weight),'REPLACE')
        groups['Hand.'+suffix].add([vertex.index],hand_weight,'REPLACE')
    axis=direction;normal=Vector((0,-1,0));side=normal.cross(axis).normalized()
    rotation=Matrix((side,normal,axis)).transposed().to_quaternion()
    pad_center=hand+normal*.149-axis*.015
    pad=surface_feature('PawPad.'+suffix,pad_center,(.099,.012,.104),surfaces[1],'Hand.'+suffix,rotation,40,24)
    for vertex in pad.data.vertices:
        local=rotation.inverted()@(vertex.co-pad_center)
        local.x*=.92+.16*local.z/.104
        vertex.co=pad_center+rotation@local
    for digit in range(2):
        point=hand+axis*.135+side*((digit-.5)*.175)+normal*.13
        surface_feature('DigitPad.'+suffix+str(digit),point,(.034,.01,.034),surfaces[1],'Hand.'+suffix,rotation,24,16)
    return shoulder,elbow,wrist,hand+direction*.20


def make_rig(definitions):
    data=bpy.data.armatures.new('MoriiReconstructionSkeleton')
    rig=bpy.data.objects.new('MoriiReconstructionRig',data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for name,start,end,parent in definitions:
        bone=data.edit_bones.new(name)
        bone.head=start; bone.tail=end
        if parent:bone.parent=data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in PARTS:
        obj.parent=rig
        modifier=obj.modifiers.new('ReconstructionSkin','ARMATURE'); modifier.object=rig
    for bone in rig.pose.bones:bone.rotation_mode='QUATERNION'
    return rig


def rotate(rig,name,axis,angle):
    bone=rig.pose.bones[name]
    rest=bone.bone.matrix_local.to_quaternion()
    bone.rotation_quaternion=rest.inverted()@Quaternion(axis,angle)@rest


def aim_arm(rig,suffix,elbow_pixel,wrist_pixel,hand_up=(0,0,1),normal=(0,-1,0)):
    shoulder=rig.pose.bones['Arm.'+suffix].head.copy()
    elbow=pixel(elbow_pixel,-0.36 if normal[1]>0 else -.10)
    wrist=pixel(wrist_pixel,-0.45 if normal[1]>0 else -.15)
    # Static pose calibration follows visible joints, a production IK rig is a later approval step.
    for name,start,target in [('Arm.'+suffix,shoulder,elbow),('Forearm.'+suffix,elbow,wrist)]:
        bone=rig.pose.bones[name]
        orientation=(target-start).normalized().to_track_quat('Y','Z').to_matrix().to_4x4()
        orientation=orientation@Matrix.Diagonal((1,(target-start).length/bone.bone.length,1,1))
        orientation.translation=start
        bone.matrix=orientation
        bpy.context.view_layer.update()
    bone=rig.pose.bones['Hand.'+suffix]
    direction=Vector(hand_up).normalized()
    front=(Vector(normal)-direction*Vector(normal).dot(direction)).normalized()
    # Preserve the rest palm normal when aligning the long axis to the reference pose.
    side=direction.cross(front).normalized()
    target=Matrix((side,direction,front)).transposed()
    rest=rig.data.bones['Hand.'+suffix]
    rest_axis=(rest.tail_local-rest.head_local).normalized()
    rest_front=Vector((0,-1,0))
    rest_front=(rest_front-rest_axis*rest_front.dot(rest_axis)).normalized()
    rest_side=rest_axis.cross(rest_front).normalized()
    source=Matrix((rest_side,rest_axis,rest_front)).transposed()
    matrix=(target@source.inverted()@rest.matrix_local.to_3x3()).to_4x4()
    matrix.translation=wrist
    bone.matrix=matrix
    bpy.context.view_layer.update()


def pose(rig,name):
    for bone in rig.pose.bones:
        bone.matrix_basis=Matrix.Identity(4)
    calibration={
        'wave':((644,364),-10.8,(0,0,0)),
        'listen':((685.368,395.578),-18.637,(.16,0,0)),
        'think':((664.896,390.559),-16.906,(.12,0,0)),
        'explain':((627.996,361.502),-11.054,(0,0,0)),
        'encourage':((642.526,344.487),-9.300,(0,0,0)),
    }
    origin,angle,translation=calibration[name]
    rig.pose.bones['Root'].matrix=Matrix.Translation(translation)@rig.data.bones['Root'].matrix_local
    if name in ('listen','think'):
        rig.pose.bones['Spine'].matrix=Matrix.Translation((translation[0],0,-.10))@rig.data.bones['Spine'].matrix_local
    bpy.context.view_layer.update()
    head_bone=rig.pose.bones['Head']
    matrix=Matrix.Rotation(math.radians(angle),4,'Y')@rig.data.bones['Head'].matrix_local
    matrix.translation=pixel(origin)
    head_bone.matrix=matrix
    bpy.context.view_layer.update()
    tail_offset={'wave':(0,0,0),'listen':(-.03,0,.003),'think':(-.04,0,.026),'explain':(-.10,0,0),'encourage':(-.04,0,.055)}[name]
    rig.pose.bones['Tail'].matrix=Matrix.Translation(tail_offset)@rig.data.bones['Tail'].matrix_local
    bpy.context.view_layer.update()
    if name in ('wave','explain'):
        aim_arm(rig,'R',(501,733),(548,785) if name=='wave' else (601,765),(.82,0,-.57) if name=='wave' else (.97,0,.24),(0,1,0))
        aim_arm(rig,'L',(860,678) if name=='wave' else (884,645),(939,628) if name=='wave' else (999,615),(.38,0,.925) if name=='wave' else (.94,0,.3),(0,-1,0))
    elif name=='encourage':
        aim_arm(rig,'R',(440,664),(315,618),(-.4,0,.916),(0,-1,0))
        aim_arm(rig,'L',(901,647),(975,583),(.4,0,.916),(0,-1,0))
    elif name in ('think','listen'):
        aim_arm(rig,'R',(566,783) if name=='think' else (578,826),(665,674) if name=='think' else (636,879),(.46,0,.888) if name=='think' else (.73,0,-.683),(0,1,0))
        aim_arm(rig,'L',(884,784) if name=='think' else (951,650),(791,810) if name=='think' else (975,578),(-.83,0,-.558) if name=='think' else (.24,0,.971),(0,1,0) if name=='think' else (0,-1,0))
    if name=='think':
        align_foot(rig,'R',(561,1045),(-.6,0,.8),(0,-1,0))
        align_foot(rig,'L',(955,1017),(.65,0,.76),(0,-1,0))
    elif name=='encourage':
        align_foot(rig,'R',(478,1140),(0,-1,.03),(0,0,-1))
        align_foot(rig,'L',(867,1135),(0,-1,.03),(0,0,-1))
    for obj_name in ('Eye.R','Eye.L','Highlight.R','Highlight.L'):
        obj=bpy.data.objects[obj_name]
        if obj.data.shape_keys and 'look_up' in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks['look_up'].value=1 if name=='think' else 0
    obj=bpy.data.objects['MoriiHeadSurface']
    if obj.data.shape_keys and 'open_smile' in obj.data.shape_keys.key_blocks:
        obj.data.shape_keys.key_blocks['open_smile'].value=1 if name=='encourage' else 0
    bpy.context.view_layer.update()
    correct_arm_surface(rig,name)
    correct_body_surface(rig,name)


def align_foot(rig,suffix,center_pixel,direction,sole_normal):
    # A calibrated sole orientation uses the existing foot surface, not a new model for seated poses.
    rest_center=Vector((.075+(.43 if suffix=='L' else -.43),-.115,.14))
    axis=Vector((0,-1,.03)).normalized();normal=Vector((0,0,-1))
    normal=(normal-axis*axis.dot(normal)).normalized()
    source=Matrix((normal.cross(axis),normal,axis)).transposed()
    axis=Vector(direction).normalized();normal=Vector(sole_normal).normalized()
    target=Matrix((normal.cross(axis),normal,axis)).transposed()
    rotation=target@source.inverted()
    bone=rig.pose.bones['Foot.'+suffix];rest=bone.bone
    matrix=(rotation@rest.matrix_local.to_3x3()).to_4x4()
    matrix.translation=pixel(center_pixel,-.22)-rotation@(rest_center-rest.head_local)
    bone.matrix=matrix
    bpy.context.view_layer.update()


def expression_surfaces(surfaces):
    # Static reference expressions are geometry review keys, animation timing awaits approval.
    for name in ('Eye.R','Eye.L','Highlight.R','Highlight.L'):
        obj=bpy.data.objects[name];obj.shape_key_add(name='Basis')
        key=obj.shape_key_add(name='look_up')
        for vertex in key.data:vertex.co+=Vector((.014,0,.030))
    open_smile(surfaces)
    for suffix,sign in [('L',1),('R',-1)]:
        center=Vector((.075+sign*.43,-.10,.001))
        rotation=Matrix.Rotation(math.pi/2,3,'X').to_quaternion()
        surface_feature('FootPad.'+suffix,center,(.140,.006,.14),surfaces[1],'Foot.'+suffix,rotation,48,24)
        for index in range(3):
            point=center+Vector(((index-1)*.09,-.13,0))
            surface_feature('FootDigitPad.'+suffix+str(index),point,(.042,.005,.042),surfaces[1],'Foot.'+suffix,rotation,32,20)


def open_smile(surfaces):
    # The mouth recess follows the existing smile corners and the reference's rounded lower lip.
    pixels=[(558,457),(586,472),(625,470),(660,450),(689,458),(728,447),(754,417),(741,466),(720,493),(696,517),(662,530),(635,516),(611,493),(588,466)]
    contour=[Vector((unroll(p).x,-unroll(p).y)) for p in pixels]
    center=sum(contour,Vector((0,0)))/len(contour)
    obj=bpy.data.objects['MoriiHeadSurface']
    if obj.data.shape_keys is None:obj.shape_key_add(name='Basis')
    key=obj.data.shape_keys.key_blocks.get('open_smile') or obj.shape_key_add(name='open_smile')
    for base,vertex in zip(obj.data.shape_keys.key_blocks['Basis'].data,key.data):
        vertex.co=base.co
        local=base.co-pixel(HEAD_ORIGIN);point=Vector((local.x,local.z))
        inside=False;distance=10
        for index,start in enumerate(contour):
            end=contour[(index+1)%len(contour)]
            edge=end-start
            projection=start+edge*max(0,min(1,(point-start).dot(edge)/edge.length_squared))
            distance=min(distance,(point-projection).length)
            if (start.y>point.y)!=(end.y>point.y) and point.x<(end.x-start.x)*(point.y-start.y)/(end.y-start.y)+start.x:inside=not inside
        if inside and local.y<-.3:
            amount=min(1,distance/.025);amount=amount*amount*(3-2*amount)
            vertex.co.y+=.16*amount
    vertices=[];faces=[]
    radial,rings=96,20
    vertices.append(head_position(center.x,center.y,head_depth(center.x,center.y)+.08))
    for ring in range(1,rings+1):
        for segment in range(radial):
            edge=catmull(contour,segment/radial,True)
            point=center+(edge-center)*ring/rings
            vertices.append(head_position(point.x,point.y,head_depth(point.x,point.y)+.08))
        for segment in range(radial):
            a=1+(ring-1)*radial+segment;b=1+(ring-1)*radial+(segment+1)%radial
            faces.append((0,b,a) if ring==1 else (a-radial,b-radial,b,a))
    mouth=mesh('MouthInterior',vertices,faces,[material('StudyMouth',(101,51,43),.72)],'Head')
    center=face_point((662,506))
    tongue=surface_feature('Tongue',center,(.13,.008,.075),material('StudyTongue',(227,132,130),.65),'Head',segments=64,rings=32)
    for vertex in tongue.data.vertices:
        local=vertex.co-pixel(HEAD_ORIGIN)
        vertex.co.y=head_depth(local.x,local.z)+.059+(vertex.co.y-center.y)*.3


def attach_sole_pads():
    obj=bpy.data.objects['MoriiContinuousBody']
    tree=BVHTree.FromPolygons([v.co.copy() for v in obj.data.vertices],[list(p.vertices) for p in obj.data.polygons])
    for pad in bpy.context.scene.objects:
        if pad.name.startswith(('FootPad.','FootDigitPad.')):
            for vertex in pad.data.vertices:
                location,normal,face,distance=tree.ray_cast(Vector((vertex.co.x,vertex.co.y,-.5)),Vector((0,0,1)))
                if location is not None:vertex.co.z=location.z-.004+(vertex.co.z-.001)*.3


def correct_arm_surface(rig,pose_name):
    # A static reference-pose corrective preserves the shared topology while skinning is reviewed.
    for suffix,sign in [('L',1),('R',-1)]:
        obj=bpy.data.objects['ArmSurface.'+suffix]
        for modifier in list(obj.modifiers):
            if modifier.type=='ARMATURE':obj.modifiers.remove(modifier)
        if obj.data.shape_keys is None:obj.shape_key_add(name='Basis')
        keys=obj.data.shape_keys.key_blocks
        name='reference_'+pose_name
        key=keys.get(name) or obj.shape_key_add(name=name)
        for block in keys:
            if block.name!='Basis':block.value=0
        upper=rig.pose.bones['Arm.'+suffix]
        forearm=rig.pose.bones['Forearm.'+suffix]
        wrist=rig.pose.bones['Hand.'+suffix]
        rest=wrist.bone
        rest_axis=(rest.tail_local-rest.head_local).normalized()
        center=(wrist.matrix@rest.matrix_local.inverted())@(rest.head_local+rest_axis*.145)
        direction=(wrist.tail-wrist.head).normalized()
        root=rig.pose.bones['Spine'].matrix@rig.data.bones['Spine'].matrix_local.inverted()@Vector((.08+sign*.29,.04,1.52))
        points=[root,upper.head,forearm.head,wrist.head,center,center+direction*.23]
        radii=[.025,.18,.20,.145,.205,.001]
        radial=48
        rings=len(obj.data.vertices)//radial-1
        for ring in range(rings+1):
            t=ring/rings
            position=catmull(points,t)
            tangent=(catmull(points,min(1,t+.001))-catmull(points,max(0,t-.001))).normalized()
            side=tangent.cross(Vector((0,1,0))).normalized()
            depth=side.cross(tangent).normalized()
            radius=max(.001,catmull([(value,0,0) for value in radii],t).x)
            if t>=.6:
                amount=(t-.8)/.2
                curvature=1 if amount>=0 else .48
                radius=max(.001,.205*math.sqrt(max(0,1-curvature*amount**2)))
            for segment in range(radial):
                angle=math.tau*segment/radial
                coordinate=position+radius*(side*math.cos(angle)+depth*math.sin(angle)*.74)
                if t>.88:
                    local=(coordinate-center).dot(side)
                    notch=sum(math.exp(-((local-offset)/.014)**2) for offset in (-.07,.07))
                    coordinate-=direction*.015*notch*min(1,(t-.88)/.08)
                key.data[ring*radial+segment].co=coordinate
        key.value=1
        obj['review_deformation']='Static shared-topology corrective, production skinning and transitions require approval first'


def correct_body_surface(rig,pose_name):
    # Static review keys avoid the previous hard skin-weight seam, they are not final animation skinning.
    obj=bpy.data.objects['MoriiContinuousBody']
    for modifier in list(obj.modifiers):
        if modifier.type=='ARMATURE':obj.modifiers.remove(modifier)
    if obj.data.shape_keys is None:obj.shape_key_add(name='Basis')
    keys=obj.data.shape_keys.key_blocks
    key=keys.get('reference_'+pose_name) or obj.shape_key_add(name='reference_'+pose_name)
    for block in keys:
        if block.name!='Basis':block.value=0
    for index,vertex in enumerate(keys['Basis'].data):
        point=vertex.co.copy();position=point.copy()
        if pose_name in ('listen','think'):
            position.x+=.11
            position.z-=.10+.15*math.exp(-((point.z-.4)/.35)**2)*math.exp(-((point.x-.16)/.5)**4)
            lateral=min(1,max(0,(abs(point.x-.16)-.12)/.25))
            position.x+=(.08 if pose_name=='listen' else .045)*(1 if point.x>.16 else -1)*lateral*math.exp(-((point.z-.6)/.35)**2)
            foot_weight=min(1,max(0,(.43-point.z)/.17))
            foot_weight=foot_weight*foot_weight*(3-2*foot_weight)
            if pose_name=='think':
                suffix='L' if point.x>.075 else 'R'
                bone=rig.pose.bones['Foot.'+suffix]
                foot=bone.matrix@bone.bone.matrix_local.inverted()@point
            else:foot=point+Vector((.16,0,0))
            position=position.lerp(foot,foot_weight)
        elif pose_name=='encourage':
            amount=min(1,max(0,(.64-point.z)/.35))
            position.x+=(.115 if point.x>.075 else -.09)*amount
            position.z-=.04*amount
            suffix='L' if point.x>.075 else 'R'
            bone=rig.pose.bones['Foot.'+suffix]
            foot=bone.matrix@bone.bone.matrix_local.inverted()@point
            foot_weight=min(1,max(0,(.43-point.z)/.17))
            foot_weight=foot_weight*foot_weight*(3-2*foot_weight)
            position=position.lerp(foot,foot_weight)
        key.data[index].co=position
    key.value=1
    obj['review_deformation']='Shared static body pose keys, final weights and transition checks require approval'


def build():
    for directory in (VALIDATION/'renders',SOURCE):directory.mkdir(parents=True,exist_ok=True)
    previous=bpy.context.scene
    scene=bpy.data.scenes.new('Morii Likeness Reconstruction')
    bpy.context.window.scene=scene
    surfaces=[material('StudyTurquoise',(112,184,181),.74),material('StudyCream',(248,233,213),.78),material('StudyPink',(250,174,137),.8),material('StudyCocoa',(66,41,31),.44),material('StudyEyes',(44,25,21),.27),material('StudyHighlight',(255,250,243),.25)]
    for surface in surfaces[:3]:plush_surface(surface)
    head_obj=head(surfaces)
    facial_features(surfaces)
    expression_surfaces(surfaces)
    ear('R',(346,316),surfaces);ear('L',(850,224),surfaces)
    tufts(surfaces)
    torso(surfaces)
    definitions=[('Root',(0,0,0),(0,0,.2),None),('Spine',(.08,.04,.46),(.02,.04,1.70),'Root'),('Head',pixel(HEAD_ORIGIN),(pixel(HEAD_ORIGIN)+Vector((0,0,.7))),'Spine'),('Tail',(.08,.19,.5),(-.65,.22,.55),'Root'),('TailTip',(-.65,.22,.55),(-1.14,.19,.86),'Tail')]
    for suffix,sign in [('L',1),('R',-1)]:
        a,b,c,d=arm(suffix,surfaces)
        definitions.extend([('Arm.'+suffix,a,b,'Spine'),('Forearm.'+suffix,b,c,'Arm.'+suffix),('Hand.'+suffix,c,d,'Forearm.'+suffix),('Leg.'+suffix,(sign*.31+.08,.04,.57),(sign*.43+.075,.015,.18),'Root'),('Foot.'+suffix,(sign*.43+.075,.015,.18),(sign*.43+.075,-.29,.12),'Leg.'+suffix)])
        tube('LegSurface.'+suffix,[(sign*.32+.08,.04,.64),(sign*.38+.08,.03,.42),(sign*.43+.075,.015,.19)],[.23,.25,.19],surfaces[0],'Leg.'+suffix,segments=32)
        foot=paw('FootSurface.'+suffix,(sign*.43+.075,-.115,.14),(0,-1,.03),(0,0,1),surfaces[0],'Foot.'+suffix,True)[0]
        center=Vector((sign*.43+.075,-.115,.14))
        for vertex in foot.data.vertices:
            point=vertex.co-center;point.x*=1.2;point.y*=1.85;vertex.co=center+point
    fuse_body(surfaces)
    attach_sole_pads()
    tail(surfaces)
    rig=make_rig(definitions)
    references=bpy.data.collections.new('Authoritative PNG references')
    scene.collection.children.link(references)
    for name in ('wave','listen','explain','think','encourage'):
        image=bpy.data.images.load(str(ROOT/'exports'/'images'/('morii-'+name+'.png')),check_existing=True)
        ref=bpy.data.objects.new('Reference '+name,None)
        references.objects.link(ref);ref.empty_display_type='IMAGE';ref.data=image
        ref.empty_display_size=1254/PPM;ref.empty_image_depth='BACK';ref.color[3]=.5
        ref.location=((627-640)/PPM,.9,(1164-627)/PPM);ref.rotation_euler=(math.pi/2,0,0);ref.hide_render=True;ref.hide_viewport=name!='wave'
        ref['authoritative_path']=str(ROOT/'exports'/'images'/('morii-'+name+'.png'))
    camera_data=bpy.data.cameras.new('Calibrated reference camera')
    camera=bpy.data.objects.new('Calibrated reference camera',camera_data);scene.collection.objects.link(camera)
    camera.location=((627-640)/PPM,-8,(1164-627)/PPM)
    camera.rotation_euler=(Vector((camera.location.x,0,camera.location.z))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera_data.type='ORTHO';camera_data.ortho_scale=1254/PPM;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True
    scene.render.resolution_x=1254;scene.render.resolution_y=1254;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.film_transparent=True
    scene.view_settings.view_transform='Khronos PBR Neutral'
    world=bpy.data.worlds.new('Reconstruction studio');world.use_nodes=True;scene.world=world
    background=next(node for node in world.node_tree.nodes if node.type=='BACKGROUND')
    background.inputs['Color'].default_value=(.85,.85,.85,1);background.inputs['Strength'].default_value=.4
    for name,location,power,size in [('Key',(-3,-5,6),357.5,5),('Fill',(4,-3,3),260,4),('Rim',(0,3,5),227.5,4)]:
        light_data=bpy.data.lights.new('Study '+name,'AREA');light_data.energy=power;light_data.shape='DISK';light_data.size=size
        light=bpy.data.objects.new('Study '+name,light_data);scene.collection.objects.link(light);light.location=location
        light.rotation_euler=(Vector((0,0,1.7))-light.location).to_track_quat('-Z','Y').to_euler()
    pose(rig,'wave')
    scene['reconstruction_status']='Geometry review candidate, not approved, no runtime replacement'
    scene['camera_assumption']='Orthographic frontal approximation, 350 pixels/metre, original PNG canvas unchanged'
    bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'morii-candidate.blend'))
    print('MORII_RECONSTRUCTION',len(PARTS),'parts',sum(len(obj.data.vertices) for obj in PARTS),'vertices',scene.name)


if __name__=='__main__':build()
