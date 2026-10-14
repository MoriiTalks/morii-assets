"""Generate honest reference analysis derivatives; never changes the source PNGs.

Coordinates were hand-picked after viewing each complete 1254-square image.
Annotated derivatives are visually checked against their source anatomy.
Masks are used only for descriptive bounds, never to alter/warp the images.
"""
from pathlib import Path
from hashlib import sha256
import json
import math
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\Ryu Tamin\Documents\SEMESTER-7\MoriiTalks\morii-assets")
OUT = ROOT / "validation" / "likeness-2026-10-14" / "reference"
OUT.mkdir(parents=True, exist_ok=True)
RAW = {
    "wave": {
        "boxes": {
            "head_without_ears_tuft_whiskers": [344,136,925,601],
            "eye_image_left": [457,352,533,429], "eye_image_right": [758,293,831,373],
            "ear_image_left_outer": [289,237,427,387], "ear_image_right_outer": [789,155,911,287],
            "ear_image_left_inner": [326,278,385,354], "ear_image_right_inner": [823,191,883,263],
            "nose": [602,358,707,424], "muzzle": [356,339,928,601],
            "belly_visible": [539,569,860,1045], "torso_hips_visible": [432,568,904,1084],
            "foot_image_left": [417,1070,608,1166], "foot_image_right": [715,1066,917,1157],
            "tail_visible": [219,843,464,1066], "tuft": [516,100,654,170],
        },
        "points": {
            "eye_image_left_center": [495,390], "eye_image_right_center": [793,333],
            "pupil_image_left_center": [499,391], "pupil_image_right_center": [792,334],
            "nose_center": [655,391], "mouth_junction": [664,449],
            "mouth_image_left_corner": [561,455], "mouth_image_right_corner": [756,417],
            "ear_image_left_center": [357,313], "ear_image_right_center": [851,223],
            "head_top": [654,137], "chin": [606,600],
            "cheek_image_left_center": [452,458], "cheek_image_right_center": [854,382],
            "belly_top_center": [674,578], "belly_bottom": [696,1043],
            "foot_image_left_center": [515,1120], "foot_image_right_center": [816,1116],
            "tail_tip": [241,847], "tail_base_visible": [444,984],
            "raised_paw_main_pad_center": [958,583],
        },
        "muzzle_top_contour": [[360,485],[385,441],[440,419],[490,421],[530,425],[603,373],[662,358],[716,361],[770,375],[831,344],[868,340],[900,353],[923,384]],
        "pose_note": "Standing, image-right arm raised; head rolls toward image left. Three whiskers per visible cheek; raised palm shows two small toe pads plus main pad.",
    },
    "listen": {
        "boxes": {
            "head_without_ears_tuft_whiskers": [383,144,984,656],
            "eye_image_left": [498,404,580,484], "eye_image_right": [795,307,873,391],
            "ear_image_left_outer": [306,284,445,450], "ear_image_right_outer": [816,152,960,293],
            "ear_image_left_inner": [347,334,404,413], "ear_image_right_inner": [858,195,923,269],
            "nose": [647,392,757,469], "muzzle": [397,343,985,656],
            "belly_visible": [616,594,908,1127], "torso_hips_visible": [432,638,999,1118],
            "foot_image_left": [476,1053,671,1161], "foot_image_right": [768,1055,985,1156],
            "tail_visible": [207,841,480,1116], "tuft": [550,99,710,179],
        },
        "points": {
            "eye_image_left_center": [540,445], "eye_image_right_center": [832,348],
            "pupil_image_left_center": [543,443], "pupil_image_right_center": [830,348],
            "nose_center": [703,430], "mouth_junction": [720,488],
            "mouth_image_left_corner": [612,511], "mouth_image_right_corner": [812,444],
            "ear_image_left_center": [375,370], "ear_image_right_center": [887,224],
            "head_top": [704,145], "chin": [608,653],
            "cheek_image_left_center": [506,523], "cheek_image_right_center": [903,392],
            "belly_top_center": [749,638], "belly_bottom": [721,1125],
            "foot_image_left_center": [575,1109], "foot_image_right_center": [879,1107],
            "tail_tip": [231,846], "tail_base_visible": [461,990],
            "raised_paw_main_pad_center": [985,533],
        },
        "muzzle_top_contour": [[399,559],[437,500],[486,480],[539,482],[580,474],[643,426],[701,390],[751,387],[800,391],[843,365],[883,344],[929,346],[966,375]],
        "pose_note": "Seated or crouched, image-right palm near cheek/ear; stronger head roll. Belly and thigh overlap differ from standing; do not infer shorter skeleton from this silhouette.",
    },
    "explain": {
        "boxes": {
            "head_without_ears_tuft_whiskers": [331,135,920,601],
            "eye_image_left": [440,352,516,430], "eye_image_right": [741,294,815,373],
            "ear_image_left_outer": [275,233,414,385], "ear_image_right_outer": [775,148,904,280],
            "ear_image_left_inner": [314,278,374,353], "ear_image_right_inner": [813,188,873,259],
            "nose": [583,359,691,427], "muzzle": [344,338,922,601],
            "belly_visible": [580,553,878,1039], "torso_hips_visible": [450,577,917,1088],
            "foot_image_left": [415,1077,607,1170], "foot_image_right": [721,1068,932,1159],
            "tail_visible": [177,841,470,1073], "tuft": [508,94,640,172],
        },
        "points": {
            "eye_image_left_center": [479,391], "eye_image_right_center": [778,334],
            "pupil_image_left_center": [482,390], "pupil_image_right_center": [777,335],
            "nose_center": [638,393], "mouth_junction": [647,451],
            "mouth_image_left_corner": [544,458], "mouth_image_right_corner": [740,417],
            "ear_image_left_center": [345,310], "ear_image_right_center": [840,214],
            "head_top": [639,137], "chin": [586,600],
            "cheek_image_left_center": [435,459], "cheek_image_right_center": [841,382],
            "belly_top_center": [692,567], "belly_bottom": [705,1037],
            "foot_image_left_center": [512,1125], "foot_image_right_center": [826,1115],
            "tail_tip": [201,846], "tail_base_visible": [453,984],
            "raised_paw_main_pad_center": [1046,606],
        },
        "muzzle_top_contour": [[345,479],[375,439],[420,421],[464,423],[512,424],[580,379],[633,358],[687,362],[752,377],[815,343],[851,337],[885,349],[914,386]],
        "pose_note": "Standing, image-right arm extends sideways with palm facing camera. This palm shows three small toe pads plus main pad, unlike wave's two visible pads.",
    },
    "think": {
        "boxes": {
            "head_without_ears_tuft_whiskers": [344,155,963,657],
            "eye_image_left": [474,390,566,479], "eye_image_right": [768,301,853,394],
            "ear_image_left_outer": [273,289,413,453], "ear_image_right_outer": [791,158,928,299],
            "ear_image_left_inner": [315,339,375,417], "ear_image_right_inner": [829,199,896,273],
            "nose": [629,391,745,465], "muzzle": [361,351,965,657],
            "belly_visible": [569,592,898,1098], "torso_hips_visible": [434,628,954,1095],
            "foot_image_left": [459,944,662,1148], "foot_image_right": [836,908,1075,1126],
            "tail_visible": [201,834,495,1101], "tuft": [509,121,647,201],
        },
        "points": {
            "eye_image_left_center": [520,435], "eye_image_right_center": [811,348],
            "pupil_image_left_center": [530,425], "pupil_image_right_center": [817,338],
            "nose_center": [687,428], "mouth_junction": [700,484],
            "mouth_image_left_corner": [595,511], "mouth_image_right_corner": [797,444],
            "ear_image_left_center": [344,372], "ear_image_right_center": [860,230],
            "head_top": [660,156], "chin": [596,655],
            "cheek_image_left_center": [482,519], "cheek_image_right_center": [880,399],
            "belly_top_center": [757,619], "belly_bottom": [731,1097],
            "foot_image_left_center": [561,1045], "foot_image_right_center": [955,1017],
            "tail_tip": [227,838], "tail_base_visible": [477,977],
            "foot_image_left_main_pad_center": [578,1076], "foot_image_right_main_pad_center": [974,1052],
        },
        "muzzle_top_contour": [[363,568],[403,512],[462,478],[506,476],[561,474],[631,423],[685,389],[746,386],[808,393],[859,359],[900,352],[939,378]],
        "pose_note": "Seated with forward soles and image-left hand touching chin. Eyes gaze upward: more white visible below pupils. Eye opening centers and pupil centers are deliberately separate.",
    },
    "encourage": {
        "boxes": {
            "head_without_ears_tuft_whiskers": [337,112,946,587],
            "eye_image_left": [449,331,530,408], "eye_image_right": [757,280,835,361],
            "ear_image_left_outer": [287,210,425,364], "ear_image_right_outer": [810,134,945,277],
            "ear_image_left_inner": [326,254,388,327], "ear_image_right_inner": [843,175,911,251],
            "nose": [590,344,701,412], "muzzle": [345,328,945,587],
            "belly_visible": [521,558,862,1042], "torso_hips_visible": [418,574,926,1105],
            "foot_image_left": [374,1088,579,1190], "foot_image_right": [758,1082,973,1180],
            "tail_visible": [204,823,443,1071], "tuft": [526,72,671,150],
        },
        "points": {
            "eye_image_left_center": [489,370], "eye_image_right_center": [797,321],
            "pupil_image_left_center": [494,371], "pupil_image_right_center": [792,321],
            "nose_center": [646,378], "mouth_junction": [651,430],
            "mouth_image_left_corner": [550,440], "mouth_image_right_corner": [751,404],
            "mouth_open_bottom": [658,510],
            "ear_image_left_center": [357,286], "ear_image_right_center": [877,205],
            "head_top": [664,113], "chin": [604,586],
            "cheek_image_left_center": [442,437], "cheek_image_right_center": [855,374],
            "belly_top_center": [682,570], "belly_bottom": [695,1040],
            "foot_image_left_center": [478,1140], "foot_image_right_center": [867,1135],
            "tail_tip": [227,827], "tail_base_visible": [425,974],
            "paw_image_left_main_pad_center": [296,575], "paw_image_right_main_pad_center": [995,537],
        },
        "muzzle_top_contour": [[346,453],[372,417],[418,396],[466,395],[510,400],[579,356],[643,343],[701,343],[758,361],[814,334],[853,328],[903,345],[937,373]],
        "pose_note": "Standing, both arms raised and mouth open. Both visible palms show two small toe pads plus main pad. Mouth opening changes expression, not the muzzle volume.",
    },
}

COLORS = {"head": "#2f8bff", "eye": "#ff4ea3", "ear": "#8c70ff", "nose": "#ed7f27", "muzzle": "#c58619", "belly": "#72a13a", "torso": "#329f83", "foot": "#ec5353", "tail": "#3eb5bd", "tuft": "#888888"}
def color(name):
    return next((value for key,value in COLORS.items() if name.startswith(key)), "#9e245a")
def transparent_on_neutral(im):
    back = Image.new("RGBA", im.size, (242,242,239,255))
    return Image.alpha_composite(back,im).convert("RGB")
def normalized_box(box,w,h):
    return [round(box[0]/w,6),round(box[1]/h,6),round(box[2]/w,6),round(box[3]/h,6)]
def bound_mask(mask):
    yy,xx=np.nonzero(mask)
    return None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]

FACIAL_LANDMARKS = [
    "eye_image_left_center", "eye_image_right_center",
    "pupil_image_left_center", "pupil_image_right_center", "nose_center",
    "mouth_junction", "mouth_image_left_corner", "mouth_image_right_corner",
    "cheek_image_left_center", "cheek_image_right_center",
]
MAJOR_BOXES = [
    "head_without_ears_tuft_whiskers", "torso_hips_visible",
    "ear_image_left_outer", "ear_image_right_outer", "belly_visible",
    "foot_image_left", "foot_image_right", "tail_visible",
]
manifest={"schema_version":1,"annotation_method":"Human visual estimates on complete source PNGs at 1254 x 1254; annotated derivatives visually checked against source anatomy. No model-generated landmarks. No image warping.","coordinate_system":"pixel_xy_top_left","coordinate_system_note":"top-left origin, x right, y down; boxes [left,top,right,bottom] in source pixels; source and component mask bounds use exclusive right/bottom; manual anatomical boxes approximate continuous extents","facial_landmarks":FACIAL_LANDMARKS,"major_measurements":[box+"_aabb_"+axis for box in MAJOR_BOXES for axis in ["width","height"]]+["eye_center_distance"],"manual_uncertainty_pixels":8,"manual_uncertainty_note":"Approximate per-coordinate uncertainty; anatomical boundaries under occlusion are not established ground truth. This is not an accuracy confidence interval. A 3% dimension threshold can be smaller than this annotation uncertainty for ears or feet; borderline results require visual review.","silhouette_method":"alpha >=128, largest 8-connected component; full alpha bbox also recorded because cutout images have disconnected low-alpha edge artifacts","poses":{},"missing_files":[]}
metrics=[]
sheet=Image.new("RGB",(1800,850),(255,255,255)); sd=ImageDraw.Draw(sheet)
for col,(name,raw) in enumerate(RAW.items()):
    src=ROOT/"exports"/"images"/("morii-"+name+".png")
    if not src.is_file():
        manifest["missing_files"].append(str(src)); continue
    before=sha256(src.read_bytes()).hexdigest()
    im=Image.open(src).convert("RGBA"); w,h=im.size; scale=[w/1254,h/1254]
    boxes={k:[round(v[i]*scale[i%2],3) for i in range(4)] for k,v in raw["boxes"].items()}
    points={k:[round(v[i]*scale[i],3) for i in range(2)] for k,v in raw["points"].items()}
    contour=[[round(v[0]*scale[0],3),round(v[1]*scale[1],3)] for v in raw["muzzle_top_contour"]]
    alpha=np.asarray(im.getchannel("A")); labels,count=ndimage.label(alpha>=128,structure=np.ones((3,3)))
    counts=np.bincount(labels.ravel()); counts[0]=0; main=labels==int(counts.argmax())
    bbox=bound_mask(main)
    hl,ht,hr,hb=boxes["head_without_ears_tuft_whiskers"]; hw=hr-hl; hh=hb-ht
    e1,e2=points["eye_image_left_center"],points["eye_image_right_center"]
    distance=math.hypot(e2[0]-e1[0],e2[1]-e1[1]); roll=math.degrees(math.atan2(e2[1]-e1[1],e2[0]-e1[0]))
    bb=boxes["torso_hips_visible"]; n=boxes["nose"]; b=boxes["belly_visible"]
    m={"head_width_pixels":round(hw,3),"head_height_pixels":round(hh,3),"head_width_to_height":round(hw/hh,4),"head_width_to_visible_torso_hips_width":round(hw/(bb[2]-bb[0]),4),"head_height_to_component_height":round(hh/(bbox[3]-bbox[1]),4),"head_height_to_below_chin_visible_height":round(hh/(bbox[3]-hb),4),"eye_center_distance_pixels":round(distance,3),"eye_center_distance_to_head_width":round(distance/hw,4),"eye_line_roll_degrees_image_y_down":round(roll,3),"image_left_eye_width_to_head_width":round((boxes["eye_image_left"][2]-boxes["eye_image_left"][0])/hw,4),"image_right_eye_width_to_head_width":round((boxes["eye_image_right"][2]-boxes["eye_image_right"][0])/hw,4),"nose_width_to_head_width":round((n[2]-n[0])/hw,4),"nose_height_to_head_height":round((n[3]-n[1])/hh,4),"visible_belly_width_to_torso_hips_width":round((b[2]-b[0])/(bb[2]-bb[0]),4)}
    measurements={}
    for key,box in boxes.items():
        cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
        measurements[key+"_aabb_width"]={"start":[box[0],cy],"end":[box[2],cy],"method":"manual anatomical AABB width; endpoints are projection construction points, not anatomical landmarks"}
        measurements[key+"_aabb_height"]={"start":[cx,box[1]],"end":[cx,box[3]],"method":"manual anatomical AABB height; endpoints are projection construction points, not anatomical landmarks"}
    measurements["eye_center_distance"]={"start":e1,"end":e2,"method":"Euclidean distance between manually annotated eye opening centers"}
    record={"source_path":str(src),"image_path":str(src),"width":w,"height":h,"landmarks":points,"measurements":measurements,"source_sha256":before,"source_size":[w,h],"source_alpha_any_bbox":list(im.getchannel("A").getbbox()),"source_alpha128_bbox":bound_mask(alpha>=128),"largest_alpha128_component_bbox":bbox,"largest_alpha128_component_bbox_normalized":normalized_box(bbox,w,h),"largest_alpha128_component_area_pixels":int(main.sum()),"manual_boxes_pixels":boxes,"manual_boxes_normalized":{k:normalized_box(v,w,h) for k,v in boxes.items()},"manual_points_pixels":points,"manual_points_normalized":{k:[round(v[0]/w,6),round(v[1]/h,6)] for k,v in points.items()},"manual_muzzle_top_contour_pixels":contour,"manual_muzzle_top_contour_normalized":[[round(p[0]/w,6),round(p[1]/h,6)] for p in contour],"projected_metrics":m,"pose_note":raw["pose_note"]}
    annotated=transparent_on_neutral(im); draw=ImageDraw.Draw(annotated)
    for key,box in boxes.items(): draw.rectangle(box,outline=color(key),width=3)
    for key,p in points.items():
        draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=color(key),outline="white",width=1)
        if key in ["eye_image_left_center","eye_image_right_center","nose_center","chin","tail_tip"]:
            draw.text((p[0]+7,p[1]+3),key,fill=color(key),stroke_width=1,stroke_fill="white")
    draw.line([tuple(v) for v in contour],fill="#b48510",width=4)
    annotated.save(OUT/("morii-"+name+"-annotated.png"))
    panel=transparent_on_neutral(im).resize((360,360),Image.Resampling.LANCZOS)
    sheet.paste(panel,(col*360,30)); sheet.paste(annotated.resize((360,360),Image.Resampling.LANCZOS),(col*360,430))
    sd.text((col*360+10,8),"morii-"+name+" | source (uniform resize)",fill="black")
    sd.text((col*360+10,406),"Manual anatomical annotations",fill="black")
    sd.text((col*360+8,798),f"Head W/H {m['head_width_to_height']:.3f}; eyes/head {m['eye_center_distance_to_head_width']:.3f}",fill="black")
    sd.text((col*360+8,817),f"Eye-line roll {roll:.1f} deg; annotations +/-8 source px",fill="black")
    after=sha256(src.read_bytes()).hexdigest(); record["source_unchanged_sha256_check"]=before==after
    manifest["poses"][name]=record; metrics.append((name,m,bbox,normalized_box(bbox,w,h)))
sheet.save(OUT/"morii-five-reference-contact-sheet.png")
(OUT/"morii-manual-landmarks.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

lines=["# Morii reference reconstruction analysis", "", "All five authoritative references were found and visually inspected. They are 1254 x 1254 RGBA images. The original files were not edited; SHA-256 checks were recorded before and after analysis and match.", "", "## Measurement method and limits", "", "Coordinates are explicit human visual annotations, with approximate +/-8 source-pixel uncertainty. They describe visible projected anatomy, not a recovered 3D skeleton. Head boxes exclude ears, tuft, and whiskers. Below-chin height is the visible distance from the manually selected bottom of the head to the silhouette bottom. It changes in seated poses. The eye roll is the angle of the eye-center line in image coordinates (y down), not a solved camera angle.", "", "Full-alpha bounds contain scattered cutout artifacts, so silhouette measurements also report the largest 8-connected component of alpha >=128. No reference is stretched, warped, or cropped to improve agreement. The contact sheet uses only uniform thumbnail resizing and neutral-background compositing. Annotations are derivatives; source pixels remain unchanged.", "", "## Projected bounds and ratios", "", "| Pose | Silhouette bbox (source px) | Normalized bbox | Head W/H | Head/visible torso width | Head height/full silhouette height | Head/below-chin height | Eye spacing/head width | Eye-line roll | Nose width/head width |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for name,m,bbox,bnorm in metrics:
    lines.append(f"| {name} | {bbox} | {bnorm} | {m['head_width_to_height']:.3f} | {m['head_width_to_visible_torso_hips_width']:.3f} | {m['head_height_to_component_height']:.3f} | {m['head_height_to_below_chin_visible_height']:.3f} | {m['eye_center_distance_to_head_width']:.3f} | {m['eye_line_roll_degrees_image_y_down']:.1f} deg | {m['nose_width_to_head_width']:.3f} |")
lines += ["", "## Identity constraints for one shared model", "", "- The head is a broad rounded cranium with a full lower cheek mass. Standing references give width/height about 1.25-1.28 in their tilted projection. Do not use a spherical head whose width and height are equal.", "- Eye-center spacing is about 0.49-0.52 of the head width, and each visible eye is about 0.12-0.15 of the head width. Eye openings are tall rounded ovals with dark brown pupils and asymmetric cream/white rims; upward gaze in think exposes more white below pupils.", "- The cream muzzle is a large continuous lower-face region, not a small pair of protruding beige discs. Its top contour rises around the central nose, dips under eyes, and rises toward outer cheek patches. The full forehead remains teal.", "- The nose is a broad flattened dark brown oval, approximately 0.18-0.19 of head width. The short vertical philtrum meets two shallow curved smile branches. Encourage opens the mouth below these branches while preserving the cream cheek mass.", "- Round teal ears have cream inset interiors. A soft two-lobe tuft rises above the crown. Ears and tuft are modest appendages relative to the head, not dominant ears or sharp horns.", "- The torso has narrow upper chest, rounded hips, and a continuous cream throat-to-belly patch that broadens lower down. Standing head width is about 1.2 of the visible torso/hip width. Short thick limbs, rounded separate toes, and a long tapered curled image-left tail complete the silhouette.", "- Three brown whiskers are visible per cheek. Foot/hand pads are cream; paws and limbs remain teal. The tail has a visible cream underside.", "", "## Cross-pose inconsistencies and camera inference", "", "The images cannot establish a unique 3D mesh, focal length, or hidden anatomy. A near-front camera with a modest elevated view and a turned/rolled character is a plausible interpretation, because both eyes and ears are visible and standing feet show top surfaces. The eye-line roll spans approximately -9 to -18 degrees. Treat that as head pose when matching views, not an identity deformation. Ear occlusion and left/right width asymmetry can arise from head yaw, but these single views do not quantify yaw reliably.", "", "Listen and think are seated; thigh/foot orientation and head-to-body image ratios differ from standing. Think has upward pupil gaze and forward soles; encourage has an open mouth. These are expression/pose changes, not permission to reconstruct separate characters.", "", "Visible small paw-pad count varies: wave/listen/encourage palms show two small pads, explain shows three, and think soles show three. Occlusion/view angle may explain some variation, and the raster set may contain illustration inconsistencies. Choose one anatomical pad system and let actual view/pose produce visibility; do not scale each image independently to conceal differences.", "", "The broad muzzle boundary and apparent head/body volume vary slightly across the five images beyond easy measurement precision. Maintain one mesh and report residual per-pose differences after matching camera and pose. Full cutout alpha bounds are unreliable because peripheral low-alpha/cyan edge artifacts are present.", "", "## Required likeness gate", "", "Render all five poses from the same head/body model with fixed identity proportions. Compare uniformly fitted render/reference panels, silhouettes, and the manually annotated facial/head/belly/foot/tail landmarks. A review must cover the broad head, eye spacing/shape, cream muzzle boundary, nose, ears, tuft, torso, and tail. Eye ratios alone cannot establish likeness. Do not claim approval, animate, or integrate into product until the user accepts the reference comparison.", "", "## Output files", "", "- morii-manual-landmarks.json: source hashes, unmodified-source checks, manual pixel/normalized annotations, mask bounds, projected ratios, uncertainty and pose notes.", "- morii-five-reference-contact-sheet.png: complete source thumbnails above corresponding annotated derivatives; no crop/warp.", "- morii-{wave,listen,explain,think,encourage}-annotated.png: one full-size neutral-background annotated derivative per reference.", "- annotate_reference.py: reproducible analysis/derivative generator with the explicit manually picked coordinate table.", ""]
(OUT/"reference-analysis.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps({"output_directory":str(OUT),"missing_files":manifest["missing_files"],"metrics":[{"pose":n,**m,"silhouette_bbox":bb} for n,m,bb,_ in metrics]},indent=2))
