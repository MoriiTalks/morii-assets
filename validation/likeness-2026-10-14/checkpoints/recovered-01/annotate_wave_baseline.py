"""Independent manual annotations of the actual pre-correction Blender render."""
from pathlib import Path
from hashlib import sha256
import json
import math
from PIL import Image, ImageDraw

BASE=Path(r"C:\Users\Ryu Tamin\Documents\SEMESTER-7\MoriiTalks\morii-assets\validation\likeness-2026-10-14")
OUT=BASE/"checkpoints"/"recovered-01"
SRC=BASE/"renders"/"wave-before-correction.png"
reference=json.loads((BASE/"reference"/"morii-manual-landmarks.json").read_text(encoding="utf-8"))
before=sha256(SRC.read_bytes()).hexdigest()
im=Image.open(SRC).convert("RGBA"); w,h=im.size
assert (w,h)==(1254,1254)
# These coordinates are hand read from the candidate render itself. Reference
# coordinates are not copied or used to place candidate points.
points={
    "eye_image_left_center":[494,392], "eye_image_right_center":[794,335],
    "pupil_image_left_center":[495,391], "pupil_image_right_center":[795,334],
    "nose_center":[652,390], "mouth_junction":[660,450],
    "mouth_image_left_corner":[559,458], "mouth_image_right_corner":[752,418],
    "cheek_image_left_center":[465,486], "cheek_image_right_center":[855,411],
}
boxes={
    "head_without_ears_tuft_whiskers":[345,139,946,598],
    "torso_hips_visible":[450,580,888,1094],
    "ear_image_left_outer":[285,247,393,384],
    "ear_image_right_outer":[794,155,911,282],
    "belly_visible":[499,592,836,1027],
    "foot_image_left":[423,1086,609,1159],
    "foot_image_right":[723,1084,912,1159],
    "tail_visible":[228,858,468,1088],
}
measurements={}
for key,b in boxes.items():
    cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
    measurements[key+"_aabb_width"]={"start":[b[0],cy],"end":[b[2],cy],"method":"independently hand annotated candidate anatomical AABB width; endpoints are projection construction points"}
    measurements[key+"_aabb_height"]={"start":[cx,b[1]],"end":[cx,b[3]],"method":"independently hand annotated candidate anatomical AABB height; endpoints are projection construction points"}
measurements["eye_center_distance"]={"start":points["eye_image_left_center"],"end":points["eye_image_right_center"],"method":"Euclidean distance between independently hand annotated eye opening centers"}
record={"image_path":str(SRC),"source_path":str(SRC),"width":w,"height":h,"source_size":[w,h],"source_sha256":before,"landmarks":points,"manual_points_pixels":points,"manual_boxes_pixels":boxes,"measurements":measurements,"unverified_features":[{"name":"tail_full_visible_extent","reason":"The image-left tapered tail is identifiable and annotated, but an additional cream flap between the legs cannot be confidently assigned to tail, belly, or another mesh from this view. Tail AABB measurements cover only the clearly identifiable image-left region; complete tail extent remains unverified.","ambiguous_region_pixels":[595,1022,679,1100]}],"manual_uncertainty_pixels":8,"annotation_notes":["Independent visual candidate annotation; reference geometry coordinates were not copied.","Torso excludes arms and flattened foot pads, and starts at its visible attachment under the chin. Some exact shoulder boundaries are occluded by the head.","No toe segmentation is visible on candidate feet.","Tail underside boundary is visibly jagged. This visual defect is not hidden by the coordinate measurements."]}
doc={"schema_version":1,"coordinate_system":"pixel_xy_top_left","annotation_method":"Independent manual visual annotation of actual saved Blender render; no correspondence prediction and no geometry-projected labels.","facial_landmarks":reference["facial_landmarks"],"major_measurements":reference["major_measurements"],"manual_uncertainty_pixels":8,"poses":{"wave":record},"checkpoint":"recovered-01-pre-correction","single_pose_baseline_only":True}
back=Image.new("RGBA",im.size,(243,243,240,255)); annotated=Image.alpha_composite(back,im).convert("RGB"); draw=ImageDraw.Draw(annotated)
for name,b in boxes.items():
    draw.rectangle(b,outline="#3385ee",width=3)
    draw.text((b[0]+3,b[1]+3),name,fill="#1454b0",stroke_width=1,stroke_fill="white")
for name,p in points.items():
    draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill="#e13d80",outline="white")
    if not name.startswith("pupil"):
        draw.text((p[0]+5,p[1]+3),name,fill="#bd155b",stroke_width=1,stroke_fill="white")
draw.rectangle((595,1022,679,1100),outline="#da7b22",width=3)
draw.text((590,1010),"Unverified cream flap",fill="#b46a15",stroke_width=1,stroke_fill="white")
annotated.save(OUT/"wave-annotated.png")
record["source_unchanged_sha256_check"]=sha256(SRC.read_bytes()).hexdigest()==before
assert all(k in points for k in reference["facial_landmarks"])
assert all(k in measurements for k in reference["major_measurements"])
(OUT/"wave-landmarks.json").write_text(json.dumps(doc,indent=2),encoding="utf-8")
print(json.dumps({"json":str(OUT/"wave-landmarks.json"),"source_sha256":before,"source_unchanged":record["source_unchanged_sha256_check"],"facial_count":len(points),"measurement_count":len(measurements),"unverified_features":record["unverified_features"]},indent=2))
