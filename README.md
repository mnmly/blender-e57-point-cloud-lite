# E57 Point Cloud Lite for Blender

Imports ASTM E57 laser scans (`.e57`) into Blender 5.2+ as native **Point Cloud** objects.
Pure Python + numpy (both ship with Blender), so one zip works on every platform.

## Use

*File > Import > E57 Point Cloud (.e57)*, or drag an `.e57` file into the 3D viewport. Each file
becomes a collection and each scan a point cloud object.

| Option | Default | |
|---|---|---|
| Point Radius | 0.01 m | `radius` attribute, used when rendering |
| Keep Every Nth Point | 1 | subsample huge scans |
| Color / Intensity / Other Fields | on | `Color` (byte color), `intensity` (0–1), plus row/column, returns, time stamps, normals and extension fields (`las:x` → `las_x`) |
| Apply Scan Pose | on | the scan's pose becomes the object rotation/location |
| Recenter | on | shifts everything so the first scan sits at the origin; the shift is stored on the scene as `e57_origin` and reused by later imports so scans stay aligned |
| Skip Invalid Points | on | drops points flagged by `cartesianInvalidState` / `sphericalInvalidState` |
| Color Material | on | unlit material ("E57 Point Color") that renders the `Color` attribute |

Points are centered per scan in float64 before being stored as float32, so georeferenced
(UTM-scale) coordinates keep sub-millimetre precision.

Colors show in *Material Preview* / *Rendered* shading. Solid mode draws point clouds in a flat
color: Workbench ignores point cloud color attributes (Blender 5.3).

## Supported

Cartesian and spherical coordinates, Float / Integer / ScaledInteger fields of any bit width,
multiple scans with poses, color limits and intensity limits. Header and XML pages are
CRC-checked. Not imported: `images2D` (panoramic/pinhole images).

## Develop

```sh
tests/fetch_data.sh   # libE57Format test corpus -> tests/data
uv run pytest         # cross-checks every field against pye57 (libE57Format)
                      # also reads *.e57 in ~/Downloads (override: E57_EXTRA_DIR)

# build the extension zip into dist/
"/Applications/Blender 5.3.app/Contents/MacOS/Blender" --factory-startup \
    --command extension build --source-dir io_import_e57 --output-dir dist
```

`io_import_e57/e57.py` is the reader and has no `bpy` dependency;
`io_import_e57/__init__.py` is the Blender operator.
