"""Validate Morii likeness on original, unwarped image canvases.

Run with local Python (Pillow, NumPy, SciPy), not inside Blender::

    python source/morii/validation.py --self-test
    python source/morii/validation.py --candidate-landmarks PATH

Landmark files use ``coordinate_system: pixel_xy_top_left`` and a ``poses``
mapping. Each pose contains ``image_path``, ``width``, ``height``, ``landmarks``
(name -> [x, y]), and ``measurements`` (name -> {start: [x, y], end: [x, y]}).
Optional ``ratios`` map names to [numerator measurement, denominator measurement].
Relative image paths resolve against the landmark file's directory. Both sets
must annotate visible image features, with matching names. ``facial_landmarks``
at file level selects required facial points; otherwise all reference points
are required. Measurements are all required. Missing evidence fails a gate.

Image/metric inputs are never resized, translated, cropped, or warped. Only the
contact sheet uses one common, uniform display scale for each reference/candidate
pair. The native comparison images preserve every original pixel coordinate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as connected_components


ROOT = Path(__file__).resolve().parents[2]
POSES = ("wave", "listen", "explain", "think", "encourage")
ALPHA_THRESHOLDS = (1, 16, 32, 64, 128, 224)
PRIMARY_ALPHA = 32


def alpha_mask(image, threshold=PRIMARY_ALPHA):
    """Threshold only source alpha, including pixels equal to the cutoff."""
    return np.asarray(image.convert("RGBA"))[:, :, 3] >= threshold


def mask_iou(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("IoU requires identical original canvas dimensions")
    union = int(np.count_nonzero(reference | candidate))
    return int(np.count_nonzero(reference & candidate)) / union if union else None


def mask_bounds(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def largest_component_bounds(mask):
    """Exclude disconnected cutout artifacts from the height denominator only."""
    labels, count = connected_components(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return mask_bounds(labels == sizes.argmax())


def landmark_delta(reference, candidate, reference_height):
    if reference_height <= 0:
        raise ValueError("Reference character height must be positive")
    reference, candidate = pixel_point(reference), pixel_point(candidate)
    dx, dy = candidate[0] - reference[0], candidate[1] - reference[1]
    pixels = math.hypot(dx, dy)
    return {"dx_px": dx, "dy_px": dy, "distance_px": pixels,
            "normalized_by_reference_height": pixels / reference_height}


def pixel_point(value, canvas=None, inclusive=False):
    if len(value) != 2:
        raise ValueError("Pixel point must contain exactly x and y")
    result = [float(coordinate) for coordinate in value]
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError("Pixel coordinates must be finite")
    if canvas:
        for coordinate, bound in zip(result, canvas):
            if coordinate < 0 or coordinate > bound or (not inclusive and coordinate == bound):
                raise ValueError(f"Pixel point {result} lies outside native canvas {canvas}")
    return result


def measurement_length(value):
    start, end = pixel_point(value["start"]), pixel_point(value["end"])
    return math.hypot(end[0] - start[0], end[1] - start[1])


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_annotations(path):
    if not path or not path.is_file():
        return {}, [f"Missing landmark file: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            return {}, [f"{path}: landmark file root must be an object"]
        if data.get("coordinate_system") != "pixel_xy_top_left":
            return {}, [f"{path}: coordinate_system must be pixel_xy_top_left"]
        if not isinstance(data.get("poses"), dict):
            return {}, [f"{path}: poses mapping is required"]
        for pose, entry in data["poses"].items():
            if not isinstance(entry, dict):
                return {}, [f"{path}: pose {pose} must be an object"]
            for field in ("landmarks", "measurements", "ratios"):
                if field in entry and not isinstance(entry[field], dict):
                    return {}, [f"{path}: pose {pose} {field} must be an object"]
            for field in ("facial_landmarks", "major_measurements"):
                if field in entry and (not isinstance(entry[field], list) or not all(isinstance(name, str) for name in entry[field])):
                    return {}, [f"{path}: pose {pose} {field} must be a list of names"]
            if "image_path" in entry and not isinstance(entry["image_path"], str):
                return {}, [f"{path}: pose {pose} image_path must be a string"]
        for field in ("facial_landmarks", "major_measurements"):
            if field in data and (not isinstance(data[field], list) or not all(isinstance(name, str) for name in data[field])):
                return {}, [f"{path}: {field} must be a list of names"]
        return data, []
    except (ValueError, OSError) as error:
        return {}, [f"Cannot read {path}: {error}"]


def resolve_image(annotation, annotation_path, fallback):
    value = annotation.get("image_path")
    if value:
        path = Path(value)
        return path if path.is_absolute() else annotation_path.parent / path
    return fallback


def read_rgba(path):
    if not path.is_file():
        return None, f"Missing image: {path}"
    try:
        with Image.open(path) as image:
            if "A" not in image.getbands():
                return None, f"Image lacks source alpha: {path}"
            return image.convert("RGBA"), None
    except (OSError, ValueError) as error:
        return None, f"Cannot read image {path}: {error}"


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def checkerboard(size):
    width, height = size
    canvas = Image.new("RGBA", size, "#f2f2f2")
    draw = ImageDraw.Draw(canvas)
    for y in range(0, height, 24):
        for x in range(0, width, 24):
            if (x // 24 + y // 24) % 2:
                draw.rectangle((x, y, x + 23, y + 23), fill="#e0e0e0")
    return canvas


def save_native_comparisons(reference, candidate, out, name):
    """Preserve native pixel coordinates in pair, overlay, and alpha difference."""
    width = reference.width + candidate.width
    height = max(reference.height, candidate.height)
    pair = checkerboard((width, height + 48))
    pair.alpha_composite(reference, (0, 48))
    pair.alpha_composite(candidate, (reference.width, 48))
    draw = ImageDraw.Draw(pair)
    draw.rectangle((0, 0, width, 47), fill="white")
    draw.text((12, 10), f"{name} | reference | native {reference.width}x{reference.height}", fill="black", font=font(22))
    draw.text((reference.width + 12, 10), f"candidate | native {candidate.width}x{candidate.height}", fill="black", font=font(22))
    pair.save(out / f"{name}-pair.png")
    if reference.size != candidate.size:
        return
    ref_half, cand_half = reference.copy(), candidate.copy()
    ref_half.putalpha(reference.getchannel("A").point(lambda value: round(value * 0.5)))
    cand_half.putalpha(candidate.getchannel("A").point(lambda value: round(value * 0.5)))
    Image.alpha_composite(ref_half, cand_half).save(out / f"{name}-overlay-50.png")
    ref_mask, cand_mask = alpha_mask(reference), alpha_mask(candidate)
    pixels = np.zeros((reference.height, reference.width, 4), dtype=np.uint8)
    pixels[ref_mask & cand_mask] = (150, 150, 150, 110)
    pixels[ref_mask & ~cand_mask] = (235, 45, 55, 255)
    pixels[cand_mask & ~ref_mask] = (35, 110, 245, 255)
    Image.fromarray(pixels).save(out / f"{name}-silhouette-diff.png")


def remove_stale_pose_comparisons(out, name):
    """A missing/newly mismatched input must not leave older evidence in place."""
    for suffix in ("pair", "overlay-50", "silhouette-diff"):
        (out / f"{name}-{suffix}.png").unlink(missing_ok=True)


def save_contact_sheet(pairs, path, title):
    cell, label, margin = 470, 56, 16
    sheet = Image.new("RGB", (cell * 2 + margin * 3, (cell + label + margin) * len(POSES) + 68), "#f9fafb")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 14), title, fill="#152236", font=font(22))
    for row, pose in enumerate(POSES):
        reference, candidate = pairs.get(pose, (None, None))
        y = 68 + row * (cell + label + margin)
        draw.text((margin, y), f"{pose}: reference", fill="black", font=font(19))
        draw.text((margin * 2 + cell, y), "candidate", fill="black", font=font(19))
        present = [image for image in (reference, candidate) if image is not None]
        scale = min(cell / max(image.width for image in present), cell / max(image.height for image in present)) if present else 1
        for column, image in enumerate((reference, candidate)):
            x = margin + column * (cell + margin)
            panel = checkerboard((cell, cell))
            if image is None:
                ImageDraw.Draw(panel).text((20, 20), "MISSING IMAGE — FAIL", fill="#ad1323", font=font(20))
            else:
                preview = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
                panel.alpha_composite(preview)
            sheet.paste(panel.convert("RGB"), (x, y + label))
    sheet.save(path)


def score_annotations(reference, candidate, reference_data, reference_height, candidate_height, reference_size, candidate_size, limits):
    points = reference.get("facial_landmarks", reference_data.get("facial_landmarks", list(reference.get("landmarks", {}))))
    facial, dimensions, ratios, errors = {}, {}, {}, []
    if not points:
        errors.append("No required reference facial landmarks")
    for name in points:
        try:
            ref_point = pixel_point(reference["landmarks"][name], reference_size)
            cand_point = pixel_point(candidate["landmarks"][name], candidate_size)
            item = landmark_delta(ref_point, cand_point, reference_height)
            item["pass"] = item["normalized_by_reference_height"] <= limits["facial"]
            facial[name] = item
        except (KeyError, TypeError, ValueError, IndexError) as error:
            facial[name] = {"pass": False, "error": f"Missing/invalid landmark {name}: {error}"}
    reference_measurements, candidate_measurements = reference.get("measurements", {}), candidate.get("measurements", {})
    required_measurements = reference.get("major_measurements", reference_data.get("major_measurements", list(reference_measurements)))
    if not reference_measurements:
        errors.append("No required reference major dimensions")
    for name in required_measurements:
        try:
            measurement = reference_measurements[name]
            for label, entry, size in (("reference", measurement, reference_size), ("candidate", candidate_measurements[name], candidate_size)):
                pixel_point(entry["start"], size, inclusive=True)
                pixel_point(entry["end"], size, inclusive=True)
            ref_length = measurement_length(measurement)
            cand_length = measurement_length(candidate_measurements[name])
            if ref_length <= 0 or candidate_height <= 0:
                raise ValueError("Lengths and character heights must be positive")
            error = abs(cand_length / ref_length - 1)
            dimensions[name] = {"reference_px": ref_length, "candidate_px": cand_length,
                                "candidate_to_reference": cand_length / ref_length, "relative_error": error,
                                "reference_over_height": ref_length / reference_height,
                                "candidate_over_height": cand_length / candidate_height,
                                "pass": error <= limits["dimensions"]}
        except (KeyError, TypeError, ValueError, IndexError) as error:
            dimensions[name] = {"pass": False, "error": f"Missing/invalid measurement {name}: {error}"}
    ratio_definitions = dict(reference.get("ratios", {}))
    for name in dimensions:
        if name.endswith("_aabb_width"):
            height_name = name.removesuffix("_width") + "_height"
            if height_name in dimensions:
                ratio_definitions.setdefault(name.removesuffix("_aabb_width") + "_aspect", [name, height_name])
    identity_ratios = {
        "head_to_torso_width": ["head_without_ears_tuft_whiskers_aabb_width", "torso_hips_visible_aabb_width"],
        "head_to_torso_height": ["head_without_ears_tuft_whiskers_aabb_height", "torso_hips_visible_aabb_height"],
        "eye_separation_to_head_width": ["eye_center_distance", "head_without_ears_tuft_whiskers_aabb_width"],
        "belly_to_torso_width": ["belly_visible_aabb_width", "torso_hips_visible_aabb_width"],
    }
    for name, operands in identity_ratios.items():
        if all(operand in dimensions for operand in operands):
            ratio_definitions.setdefault(name, operands)
    for name, operands in ratio_definitions.items():
        try:
            numerator, denominator = (dimensions[key] for key in operands)
            ref_ratio = numerator["reference_px"] / denominator["reference_px"]
            cand_ratio = numerator["candidate_px"] / denominator["candidate_px"]
            error = abs(cand_ratio / ref_ratio - 1)
            ratios[name] = {"reference": ref_ratio, "candidate": cand_ratio,
                            "relative_error": error, "pass": error <= limits["dimensions"]}
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            ratios[name] = {"pass": False, "error": f"Missing/invalid ratio {name}: {error}"}
    return {"facial_landmarks": facial, "major_dimensions": dimensions, "dimension_ratios": ratios,
            "annotation_errors": errors, "facial_pass": bool(facial) and all(item["pass"] for item in facial.values()),
            "dimensions_pass": bool(dimensions) and all(item["pass"] for item in dimensions.values()) and all(item["pass"] for item in ratios.values())}


def score_pose(pose, reference, candidate, ref_entry, cand_entry, ref_data, limits):
    errors = []
    result = {"pose": pose, "pass": False, "errors": errors}
    if reference is None or candidate is None:
        errors.append("Both original reference and genuine candidate render are required")
        return result
    result["reference_canvas"] = list(reference.size)
    result["candidate_canvas"] = list(candidate.size)
    if reference.size != candidate.size:
        errors.append("Canvas dimensions differ; no registration, scaling, or IoU performed")
        return result
    for label, entry, image in (("reference", ref_entry, reference), ("candidate", cand_entry, candidate)):
        if entry.get("width") != image.width or entry.get("height") != image.height:
            errors.append(f"{label} annotation width/height do not match native image canvas")
    ref_mask, cand_mask = alpha_mask(reference), alpha_mask(candidate)
    ref_bounds, cand_bounds = largest_component_bounds(ref_mask), largest_component_bounds(cand_mask)
    if ref_bounds is None or cand_bounds is None:
        errors.append("Empty silhouette at alpha >=32")
        return result
    result["reference_raw_bbox_alpha32"] = mask_bounds(ref_mask)
    result["candidate_raw_bbox_alpha32"] = mask_bounds(cand_mask)
    result["reference_largest_component_bbox_alpha32"] = ref_bounds
    result["candidate_largest_component_bbox_alpha32"] = cand_bounds
    result["reference_largest_component_bbox_alpha128"] = largest_component_bounds(alpha_mask(reference, 128))
    result["candidate_largest_component_bbox_alpha128"] = largest_component_bounds(alpha_mask(candidate, 128))
    result["iou_by_alpha_threshold"] = {str(threshold): mask_iou(alpha_mask(reference, threshold), alpha_mask(candidate, threshold)) for threshold in ALPHA_THRESHOLDS}
    result["iou"] = result["iou_by_alpha_threshold"][str(PRIMARY_ALPHA)]
    result["iou_pass"] = result["iou"] is not None and result["iou"] >= limits["iou"]
    result.update(score_annotations(ref_entry, cand_entry, ref_data, ref_bounds[3] - ref_bounds[1], cand_bounds[3] - cand_bounds[1], reference.size, candidate.size, limits))
    result["pass"] = not errors and not result["annotation_errors"] and result["iou_pass"] and result["facial_pass"] and result["dimensions_pass"]
    return result


def markdown_report(report):
    limits = report["limits"]
    lines = ["# Morii likeness validation", "", f"Overall result: **{'PASS' if report['pass'] else 'FAIL'}**.", "",
             "Images were scored on their full original canvases. No image registration, cropping, warping, or independent axis scaling was used.", "",
             f"Gates: raw alpha >=32 silhouette IoU >={limits['iou']:.2f}; each required facial landmark delta <={limits['facial']:.0%} of reference largest 8-connected alpha>=32 component height; each required major dimension and named ratio relative error <={limits['dimensions']:.0%}.", "",
             "Disconnected cutout pixels remain in IoU. Only the facial normalization height uses the largest connected component. Both raw and component bounds are recorded.", "",
             "Landmark values are image annotations supplied by the reference and candidate annotation files. Their provenance and stated uncertainty are recorded in metrics.json; numeric gates do not independently prove renderer provenance, creative approval, or animation correctness. Reference manual annotations have finite precision; marginal errors should be inspected on the original canvas. Reference-identical candidate paths or PNG hashes are rejected.", "",
             f"| Pose | Raw canvas | IoU >=32 | Facial <={limits['facial']:.0%} Href | Dimensions/ratios <={limits['dimensions']:.0%} | Result |", "|---|---|---:|---|---|---|"]
    for pose in POSES:
        item = report["poses"][pose]
        canvas = "match" if item.get("reference_canvas") == item.get("candidate_canvas") and item.get("reference_canvas") else "FAIL"
        iou = f"{item['iou']:.5f}" if item.get("iou") is not None else "missing"
        lines.append(f"| {pose} | {canvas} | {iou} | {'PASS' if item.get('facial_pass') else 'FAIL'} | {'PASS' if item.get('dimensions_pass') else 'FAIL'} | {'PASS' if item['pass'] else 'FAIL'} |")
    lines += ["", "## Evidence files", "", "- `../comparisons/contact-sheet.png`: five reference/render pairs; display-only common uniform scale within each pair.",
              "- `../comparisons/{pose}-pair.png`: native image pixels with a 48-pixel header added outside the image canvas.",
              "- `../comparisons/{pose}-overlay-50.png`: transparent source-over overlay; each original source alpha multiplied by 0.5.",
              "- `../comparisons/{pose}-silhouette-diff.png`: red = missing reference silhouette; blue = surplus candidate silhouette; gray = intersection.",
              "- `metrics.json`: all facial/dimension errors, alpha-threshold sensitivity, source paths and SHA-256 hashes.", ""]
    if report["errors"]:
        lines += ["## Missing or invalid evidence", ""] + [f"- {error}" for error in report["errors"]] + [""]
    for pose, item in report["poses"].items():
        lines += [f"## {pose}", ""]
        lines += [f"- {error}" for error in item.get("errors", []) + item.get("annotation_errors", [])]
        for name, point in item.get("facial_landmarks", {}).items():
            detail = point.get("error", f"delta = {point.get('normalized_by_reference_height', 0):.3%} Href")
            lines.append(f"- Facial `{name}`: {detail}; {'PASS' if point['pass'] else 'FAIL'}.")
        for group in ("major_dimensions", "dimension_ratios"):
            for name, dimension in item.get(group, {}).items():
                detail = dimension.get("error", f"relative error = {dimension.get('relative_error', 0):.3%}")
                lines.append(f"- {group} `{name}`: {detail}; {'PASS' if dimension['pass'] else 'FAIL'}.")
        sensitivity = item.get("iou_by_alpha_threshold", {})
        if sensitivity:
            lines.append("- Alpha sensitivity IoU: " + "; ".join(f"{key}: {value:.5f}" if value is not None else f"{key}: empty" for key, value in sensitivity.items()) + ".")
        lines.append("")
    if report.get("baseline"):
        lines += ["## Existing render baseline", "", "Baseline is reported separately and does not affect candidate gates.", ""]
        for pose, item in report["baseline"].items():
            lines.append(f"- {pose}: {item.get('error') or 'raw-canvas IoU = ' + format(item['iou'], '.5f')}.")
        lines.append("")
    return "\n".join(lines)


def run(args):
    output = args.output.resolve()
    comparisons, report_dir = output / "comparisons", output / "report"
    comparisons.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ref_path = args.reference_landmarks.resolve()
    cand_path = args.candidate_landmarks.resolve()
    ref_data, ref_errors = load_annotations(ref_path)
    cand_data, cand_errors = load_annotations(cand_path)
    report = {"schema_version": 1, "pass": False, "limits": {"iou": args.iou, "facial": args.facial_tolerance, "dimensions": args.dimension_tolerance},
              "policy": {"coordinate_system": "pixel_xy_top_left", "metric_transform": "identity", "primary_alpha": PRIMARY_ALPHA,
                         "facial_denominator": "reference largest 8-connected alpha>=32 component bbox height", "overlay": "source-over, each alpha times 0.5"},
              "annotation_provenance": {"reference": ref_data.get("provenance", ref_data.get("annotation_method")), "candidate": cand_data.get("provenance", cand_data.get("annotation_method"))},
              "annotation_uncertainty_px": {"reference": ref_data.get("manual_uncertainty_pixels"), "candidate": cand_data.get("manual_uncertainty_pixels")},
              "annotation_files": {"reference": str(ref_path), "candidate": str(cand_path)},
              "errors": ref_errors + cand_errors, "poses": {}, "source_files": {}}
    pairs, baseline_pairs = {}, {}
    if args.baseline_dir:
        report["baseline"] = {}
    else:
        for pose in POSES:
            remove_stale_pose_comparisons(comparisons, f"baseline-{pose}")
        (comparisons / "baseline-contact-sheet.png").unlink(missing_ok=True)
    for pose in POSES:
        remove_stale_pose_comparisons(comparisons, pose)
        ref_entry, cand_entry = ref_data.get("poses", {}).get(pose, {}), cand_data.get("poses", {}).get(pose, {})
        ref_image_path = resolve_image(ref_entry, ref_path, ref_path.parent / f"{pose}.png")
        cand_image_path = resolve_image(cand_entry, cand_path, args.candidate_dir / f"{pose}.png")
        reference, ref_error = read_rgba(ref_image_path)
        candidate, cand_error = read_rgba(cand_image_path)
        pairs[pose] = (reference, candidate)
        item = score_pose(pose, reference, candidate, ref_entry, cand_entry, ref_data, report["limits"])
        item["errors"] += [error for error in (ref_error, cand_error) if error]
        report["poses"][pose] = item
        report["source_files"][pose] = {label: {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None} for label, path in (("reference", ref_image_path), ("candidate", cand_image_path))}
        source_info = report["source_files"][pose]
        if reference is not None and candidate is not None and (ref_image_path.resolve() == cand_image_path.resolve() or source_info["reference"]["sha256"] == source_info["candidate"]["sha256"] or np.array_equal(np.asarray(reference), np.asarray(candidate))):
            item["errors"].append("Candidate is the same path, PNG SHA-256, or decoded RGBA pixels as the reference; independent renderer evidence is required")
            item["pass"] = False
        if reference is not None and candidate is not None:
            save_native_comparisons(reference, candidate, comparisons, pose)
        if args.baseline_dir:
            remove_stale_pose_comparisons(comparisons, f"baseline-{pose}")
            old_path = args.baseline_dir / f"{pose}.png"
            baseline, error = read_rgba(old_path)
            baseline_pairs[pose] = (reference, baseline)
            baseline_item = {"path": str(old_path), "sha256": file_sha256(old_path) if old_path.is_file() else None}
            if error:
                baseline_item["error"] = error
            elif reference is None:
                baseline_item["error"] = "Missing reference; baseline cannot be scored"
            elif reference.size != baseline.size:
                baseline_item["error"] = f"Native canvas mismatch {reference.size} vs {baseline.size}; no IoU calculated"
                save_native_comparisons(reference, baseline, comparisons, f"baseline-{pose}")
            else:
                baseline_item["iou"] = mask_iou(alpha_mask(reference), alpha_mask(baseline))
                if baseline_item["iou"] is None:
                    baseline_item["error"] = "Empty silhouette union"
                save_native_comparisons(reference, baseline, comparisons, f"baseline-{pose}")
            report["baseline"][pose] = baseline_item
    report["pass"] = not report["errors"] and all(item["pass"] for item in report["poses"].values())
    save_contact_sheet(pairs, comparisons / "contact-sheet.png", "Morii references and supplied candidate renders")
    if args.baseline_dir:
        save_contact_sheet(baseline_pairs, comparisons / "baseline-contact-sheet.png", "Morii references and existing render baseline")
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(report_dir / "report.md"), "iou": {pose: report["poses"][pose].get("iou") for pose in POSES}}))
    return 0 if report["pass"] else 1


class MetricTests(unittest.TestCase):
    def test_iou_of_known_overlapping_rectangles(self):
        reference, candidate = np.zeros((6, 7), dtype=bool), np.zeros((6, 7), dtype=bool)
        reference[1:4, 1:4] = True
        candidate[1:4, 2:5] = True
        self.assertEqual(mask_iou(reference, candidate), 0.5)
        self.assertIsNone(mask_iou(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)))
        with self.assertRaises(ValueError):
            mask_iou(reference, np.zeros((5, 7), dtype=bool))

    def test_alpha_threshold_ignores_rgb_and_includes_boundary(self):
        pixels = np.array([[[255, 255, 255, 31], [0, 0, 0, 32], [255, 0, 0, 255]]], dtype=np.uint8)
        self.assertEqual(alpha_mask(Image.fromarray(pixels), 32).tolist(), [[False, True, True]])
        self.assertEqual(alpha_mask(Image.fromarray(pixels), 224).tolist(), [[False, False, True]])

    def test_landmark_uses_character_height_and_euclidean_distance(self):
        result = landmark_delta([100, 200], [106, 208], 500)
        self.assertEqual(result["distance_px"], 10)
        self.assertEqual(result["normalized_by_reference_height"], 0.02)
        with self.assertRaises(ValueError):
            landmark_delta([0, 0], [1, 1], 0)

    def test_disconnected_artifacts_do_not_inflate_character_height(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        mask[0, 0] = True
        mask[29, 29] = True
        self.assertEqual(mask_bounds(mask), [0, 0, 30, 30])
        self.assertEqual(largest_component_bounds(mask), [10, 10, 20, 20])

    def test_unusable_annotation_coordinates_are_rejected(self):
        for point in ([float("nan"), 0], [float("inf"), 0], [-1, 10], [100, 10]):
            with self.assertRaises(ValueError):
                pixel_point(point, (100, 100))
        self.assertEqual(pixel_point([100, 100], (100, 100), inclusive=True), [100, 100])

    def test_malformed_evidence_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            for payload in (None, [], {"coordinate_system": "pixel_xy_top_left", "poses": {"wave": None}},
                            {"coordinate_system": "pixel_xy_top_left", "poses": {"wave": {"landmarks": []}}}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                data, errors = load_annotations(path)
                self.assertEqual(data, {})
                self.assertTrue(errors)


def main():
    default = ROOT / "validation" / "likeness-2026-10-14"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-landmarks", type=Path, default=default / "reference" / "landmarks.json")
    parser.add_argument("--candidate-landmarks", type=Path, default=default / "renders" / "landmarks.json")
    parser.add_argument("--candidate-dir", type=Path, default=default / "renders")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--iou", type=float, default=0.93)
    parser.add_argument("--facial-tolerance", type=float, default=0.02)
    parser.add_argument("--dimension-tolerance", type=float, default=0.03)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests))
        return 0 if result.wasSuccessful() else 1
    for name, value in (("iou", args.iou), ("facial-tolerance", args.facial_tolerance), ("dimension-tolerance", args.dimension_tolerance)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name} must be finite and between 0 and 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
