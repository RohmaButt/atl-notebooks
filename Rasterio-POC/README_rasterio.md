# Rasterio Geospatial Analysis Toolkit

A comprehensive Python toolkit for reading, analyzing, and processing geospatial raster data using Rasterio, including advanced feature extraction, vector-raster integration, and quality assessment workflows.

---

## 📋 Overview

This project provides a complete workflow for geospatial raster analysis, originally developed as a Google Colab notebook. It demonstrates advanced capabilities for:

- Reading and manipulating raster datasets (DEMs, satellite imagery)
- Extracting and processing geospatial features
- Vector-raster overlay analysis
- Feature quality assessment and improvement
- Advanced spatial operations (clipping, reprojecting, raster algebra)

Built on top of **Rasterio** (a Python wrapper for GDAL), this toolkit bridges the gap between raw geospatial data and meaningful analysis.

---

## 🎯 Learning Objectives

By working with this project, you will be able to:

- Read, write, and manipulate raster datasets using Rasterio
- Extract metadata and perform operations on raster bands
- Visualize raster datasets and overlay them with vector data
- Perform geospatial operations (clipping, reprojecting, raster algebra)
- Extract linear features from raster data
- Assess and improve geospatial feature quality
- Apply multi-pass snapping and connectivity-based merging algorithms

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install core libraries individually:

```bash
pip install rasterio fiona geopandas numpy matplotlib shapely
```

---

## 📦 Core Dependencies

```python
import rasterio
import rasterio.plot
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
```

**Library Purposes:**
- `rasterio`: Reading and writing raster data
- `rasterio.plot`: Plotting and visualizing raster datasets
- `geopandas`: Handling vector geospatial data
- `numpy`: Array manipulations for raster operations
- `matplotlib`: Creating visualizations
- `shapely`: Geometric operations and spatial analysis

---

## 🗂️ Project Structure

```
.
├── copy_of_rasterio.py          # Main analysis script
├── requirements.txt              # Python dependencies
├── examples.py                   # Example usage scripts
├── poc_deliverables/             # Output directory
│   ├── final_improved_features.geojson
│   └── final_improved_features.shp
├── extracted_features.geojson    # Intermediate data
├── merged_features.geojson       # Processed features
└── README.md                     # This file
```

---

## 🛠️ Key Features

### 1. **Reading Raster Data**

```python
raster_path = "https://github.com/opengeos/datasets/releases/download/raster/dem_90m.tif"
src = rasterio.open(raster_path)
```

Access raster metadata:
- Coordinate Reference System (CRS)
- Spatial resolution
- Dimensions (width, height)
- Geographic bounds
- Data types
- Affine transformation matrix

### 2. **Raster Visualization**

```python
# Basic plot
rasterio.plot.show(src)

# Custom plot with styling
fig, ax = plt.subplots(figsize=(8, 8))
rasterio.plot.show(src, cmap="terrain", ax=ax, title="Digital Elevation Model (DEM)")
plt.show()
```

### 3. **Vector-Raster Overlay**

```python
# Load and reproject vector data
gdf = gpd.read_file(dem_bounds)
gdf = gdf.to_crs(src.crs)

# Overlay on raster
fig, ax = plt.subplots(figsize=(8, 8))
rasterio.plot.show(src, cmap="terrain", ax=ax)
gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)
plt.show()
```

### 4. **Feature Extraction & Quality Assessment**

The script includes advanced workflows for:

- **Multi-pass snapping**: Adaptive tolerance-based feature alignment (10m → 100m)
- **Connectivity-based merging**: Smart grouping to prevent over-fragmentation
- **Quality scoring**: Automated assessment based on gaps, overlaps, and topology
- **Gap detection**: Identifying disconnected features with configurable thresholds

---

## 🏃 Usage Examples

### Run the Main Script

```bash
python copy_of_rasterio.py
```

### Execute Specific Examples

```bash
python examples.py 1
```

### Access in Jupyter/Colab

The original notebook is available at:
```
https://colab.research.google.com/drive/182kSIt4Xsjb8eAgt-_esZeFShvWbW1sS
```

---

## 📊 Output Files

The processing pipeline generates:

1. **GeoJSON Format**: `final_improved_features.geojson`
2. **Shapefile Format**: `final_improved_features.shp`

Both contain:
- Feature type classification
- Geometry (LineString/MultiLineString)
- Length measurements (meters)
- Quality metrics

---

## 🔍 Advanced Workflows

### Feature Improvement Pipeline

The script implements a sophisticated 5-step improvement process:

1. **Load Original Features**: Import extracted geospatial features
2. **Multi-Pass Snapping**: Apply adaptive tolerance snapping (10m, 25m, 50m, 100m)
3. **Smart Merging**: Connectivity-based consolidation to reduce fragmentation
4. **Quality Recalculation**: Assess gaps, invalids, overlaps
5. **Visualization**: Compare before/after results

### Quality Metrics

```python
Quality Score = 100 - gap_penalty - overlap_penalty - invalid_penalty + consolidation_bonus
```

Where:
- `gap_penalty`: Based on detected disconnected endpoints
- `overlap_penalty`: Penalty for overlapping features
- `invalid_penalty`: Penalty for invalid geometries
- `consolidation_bonus`: Reward for reducing fragmentation

---

## 📈 Performance Benchmarks

Example improvements from the script:

| Metric                | First Attempt | Improved | Change   |
|-----------------------|---------------|----------|----------|
| Feature count         | 2,358         | ~500     | -78.8%   |
| Snapping operations   | 37            | 200+     | +441%    |
| Estimated gaps        | 4,247         | <100     | -97.7%   |
| Quality score         | 53.6/100      | 85+/100  | +58.6%   |

---

## 🧪 Understanding the Affine Transform

The affine transformation matrix maps pixel coordinates to geographic coordinates:

```
| a  b  c |
| d  e  f |
| 0  0  1 |
```

- `a`: Pixel width (x-direction)
- `e`: Pixel height (y-direction, typically negative)
- `c`: X-coordinate of upper-left corner
- `f`: Y-coordinate of upper-left corner
- `b`, `d`: Rotation (typically zero)

---

## 🌍 Supported Data Formats

### Raster Formats
- GeoTIFF (.tif, .tiff)
- HDF (.hdf, .h5)
- NetCDF (.nc)
- IMG, GRIB, and other GDAL-supported formats

### Vector Formats (for overlay)
- GeoJSON (.geojson)
- Shapefile (.shp)
- GeoPackage (.gpkg)
- KML (.kml)

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'rasterio'`
```bash
pip install rasterio --upgrade
```

**Issue**: `GDAL_DATA not found`
```bash
# On Linux/Mac
export GDAL_DATA=$(gdal-config --datadir)

# On Windows (conda)
conda install -c conda-forge gdal
```

**Issue**: CRS mismatch between raster and vector
```python
# Reproject vector to match raster
gdf = gdf.to_crs(src.crs)
```

---

## 📚 Additional Resources

- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GDAL Documentation](https://gdal.org)
- [GeoPandas Documentation](https://geopandas.org)
- [Original Colab Notebook](https://colab.research.google.com/github/giswqs/geog-312/blob/main/book/geospatial/rasterio.ipynb)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is based on educational materials. Please check the original repository for licensing information.

---

## 🙏 Acknowledgments

- Original notebook from [geog-312](https://github.com/giswqs/geog-312)
- Built with Rasterio, GeoPandas, and the Python geospatial ecosystem
- Sample datasets from [OpenGeos](https://github.com/opengeos/datasets)

---

## ✅ Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run example: `python examples.py 1`
- [ ] Check output in `poc_deliverables/`
- [ ] Explore the main script: `copy_of_rasterio.py`

---

**Happy Mapping! 🗺️**
