# New Module Specifications for Agent-Vomit Image Processing Extensions

## Overview
Specifications for new agent-vomit modules needed to enable topographic map processing for terrain synthesis. These modules fill gaps in the current agent-vomit capabilities, focusing specifically on computer vision tasks required for cartographic analysis.

---

## Module 1: ContourDetector

### Purpose
Extract and process topographic contour lines from scanned maps, distinguishing elevation contours from other map features.

### Agent Generation Prompt
```
Create a ContourDetector module that extracts elevation contour lines from topographic maps while ignoring roads, political boundaries, and other non-elevation features. The module should:

1. Detect continuous curves representing elevation contours
2. Handle broken or faded contour lines with gap interpolation
3. Distinguish contour lines from other linear map features using context clues
4. Output structured contour data with elevation associations
5. Provide confidence scoring for detected contours

Input: RGB image of topographic map
Output: List of contour polylines with associated elevation values and confidence scores

Requirements:
- Real-time processing capability for interactive applications
- Robust to scan artifacts, shadows, and image quality variations
- Configurable sensitivity for different map styles and scales
- Memory efficient for large high-resolution map images

Include comprehensive error handling and validation of contour continuity.
```

### Expected Interface
```python
class ContourDetector:
    def __init__(self, sensitivity: float = 0.7, min_contour_length: int = 50):
        pass
    
    def detect_contours(self, map_image: torch.Tensor) -> ContourData:
        """Extract contour lines from topographic map image"""
        pass
    
    def set_elevation_range(self, min_elevation: float, max_elevation: float):
        """Set expected elevation range for contour interpretation"""
        pass
    
    def calibrate_from_legend(self, legend_region: torch.Tensor):
        """Auto-calibrate contour detection from map legend"""
        pass

@dataclass
class ContourData:
    contours: List[torch.Tensor]      # List of (N,2) coordinate arrays
    elevations: List[float]           # Elevation for each contour
    confidences: List[float]          # Detection confidence scores
    contour_interval: float           # Detected contour interval
```

### Validation Criteria
- 95%+ accuracy on USGS topographic maps
- Handle contour intervals from 1m to 100m
- Process 4K map images in <2 seconds
- Robust to 20° rotation and 0.5-2.0x scale variations

---

## Module 2: ElevationOCR

### Purpose
Extract elevation numbers, coordinate information, and scale data from topographic map text elements.

### Agent Generation Prompt
```
Create an ElevationOCR module specialized for reading numerical elevation data, coordinates, and scale information from topographic maps. The module should:

1. Detect and extract elevation numbers near contour lines and peaks
2. Read coordinate grid references and datum information
3. Parse map scale and contour interval from legends
4. Handle various font styles and orientations common in cartography
5. Associate elevation text with nearby geographic features

Input: RGB image regions containing text elements
Output: Structured elevation and coordinate data with spatial associations

Requirements:
- High accuracy for numerical text recognition
- Robust to varying text orientations and sizes
- Handle both metric and imperial units
- Parse coordinate formats (UTM, lat/long, etc.)
- Real-time processing for interactive map exploration

Include confidence scoring and validation against expected elevation ranges.
```

### Expected Interface
```python
class ElevationOCR:
    def __init__(self, units: str = "metric", coordinate_system: str = "utm"):
        pass
    
    def extract_elevations(self, map_image: torch.Tensor) -> ElevationData:
        """Extract elevation numbers and associated coordinates"""
        pass
    
    def parse_legend(self, legend_region: torch.Tensor) -> MapMetadata:
        """Extract scale, contour interval, and datum from legend"""
        pass
    
    def set_elevation_bounds(self, min_elev: float, max_elev: float):
        """Set expected elevation range for validation"""
        pass

@dataclass
class ElevationData:
    elevations: List[float]           # Detected elevation values
    positions: List[Tuple[int, int]]  # Pixel coordinates of text
    confidences: List[float]          # OCR confidence scores
    units: str                        # "meters" or "feet"

@dataclass
class MapMetadata:
    scale: str                        # e.g., "1:24000"
    contour_interval: float           # Elevation between contours
    datum: str                        # Coordinate system datum
    magnetic_declination: float       # If available
```

### Validation Criteria
- 98%+ accuracy on clearly printed elevation numbers
- Handle 0-359° text rotation
- Process map legends with 95%+ metadata extraction accuracy
- Support common map scales from 1:1,000 to 1:250,000

---

## Module 3: GeometricProcessor

### Purpose
Convert irregular contour data into regular grid heightmaps suitable for terrain synthesis.

### Agent Generation Prompt
```
Create a GeometricProcessor module that converts irregular topographic contour data into regular grid heightmaps for terrain synthesis applications. The module should:

1. Interpolate elevation values between contour lines using advanced triangulation
2. Generate smooth elevation surfaces from sparse contour data
3. Handle coordinate system transformations (geographic to cartesian)
4. Provide configurable grid resolution and interpolation methods
5. Validate and smooth interpolated data to remove artifacts

Input: Irregular contour polylines with elevation data
Output: Regular grid heightmap suitable for real-time 3D rendering and audio synthesis

Requirements:
- Multiple interpolation methods (linear, cubic, RBF)
- Efficient processing for large geographic areas
- Edge handling for map boundaries
- Memory-efficient streaming for large datasets
- Real-time preview capability for interactive applications

Include quality metrics for interpolation accuracy and surface smoothness.
```

### Expected Interface
```python
class GeometricProcessor:
    def __init__(self, interpolation_method: str = "cubic", grid_resolution: int = 512):
        pass
    
    def triangulate_contours(self, contour_data: ContourData) -> TriangulatedSurface:
        """Create triangulated surface from contour data"""
        pass
    
    def generate_heightmap(self, surface: TriangulatedSurface, 
                          bounds: BoundingBox) -> torch.Tensor:
        """Generate regular grid heightmap from triangulated surface"""
        pass
    
    def transform_coordinates(self, coords: torch.Tensor, 
                            from_crs: str, to_crs: str) -> torch.Tensor:
        """Transform between coordinate reference systems"""
        pass
    
    def smooth_surface(self, heightmap: torch.Tensor, 
                      smoothing_factor: float) -> torch.Tensor:
        """Apply smoothing to reduce interpolation artifacts"""
        pass

@dataclass
class TriangulatedSurface:
    vertices: torch.Tensor            # (N,3) XYZ coordinates
    faces: torch.Tensor               # (M,3) triangle indices
    elevation_bounds: Tuple[float, float]
    interpolation_quality: float

@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
```

### Validation Criteria
- <5% RMS error compared to known elevation data
- Generate 512x512 heightmaps in <1 second
- Handle elevation ranges from 0-9000m
- Smooth surfaces with minimal interpolation artifacts

---

## Module 4: SymbolClassifier

### Purpose
Recognize and classify topographic symbols, map features, and terrain types from map imagery.

### Agent Generation Prompt
```
Create a SymbolClassifier module that recognizes standard topographic symbols and classifies terrain features from map imagery. The module should:

1. Identify common topographic symbols (peaks, water bodies, vegetation, structures)
2. Distinguish between different line types (contours, roads, boundaries, streams)
3. Classify terrain types (forest, desert, urban, water, etc.) from color and symbol patterns
4. Parse map legends to understand symbol meanings for different map standards
5. Provide spatial location and confidence for all detected features

Input: RGB map image regions
Output: Classified features with locations and terrain type information

Requirements:
- Support multiple map standards (USGS, international)
- Handle color and black/white maps
- Robust to varying map scales and symbol sizes
- Real-time processing for interactive applications
- Extensible symbol database for custom map types

Include confidence scoring and spatial relationship analysis between features.
```

### Expected Interface
```python
class SymbolClassifier:
    def __init__(self, map_standard: str = "usgs", symbol_database: str = "standard"):
        pass
    
    def classify_symbols(self, map_image: torch.Tensor) -> SymbolData:
        """Detect and classify topographic symbols"""
        pass
    
    def identify_terrain_types(self, map_image: torch.Tensor) -> TerrainClassification:
        """Classify terrain types from map patterns"""
        pass
    
    def parse_legend_symbols(self, legend_image: torch.Tensor) -> SymbolLegend:
        """Extract symbol meanings from map legend"""
        pass
    
    def add_custom_symbol(self, symbol_image: torch.Tensor, 
                         symbol_type: str, confidence_threshold: float):
        """Add custom symbol to recognition database"""
        pass

@dataclass
class SymbolData:
    symbols: List[DetectedSymbol]
    line_features: List[LineFeature]
    terrain_regions: List[TerrainRegion]

@dataclass
class DetectedSymbol:
    symbol_type: str                  # "peak", "building", "bridge", etc.
    position: Tuple[int, int]         # Pixel coordinates
    confidence: float
    properties: Dict[str, Any]        # Symbol-specific properties

@dataclass
class LineFeature:
    feature_type: str                 # "road", "stream", "contour", "boundary"
    polyline: torch.Tensor            # (N,2) coordinate array
    confidence: float
    style_properties: Dict[str, Any]  # Line style, width, etc.

@dataclass
class TerrainRegion:
    terrain_type: str                 # "forest", "water", "urban", etc.
    boundary: torch.Tensor            # Polygon boundary
    confidence: float
    properties: Dict[str, Any]        # Vegetation density, water type, etc.
```

### Validation Criteria
- 90%+ accuracy on standard USGS symbols
- Handle symbol scales from 1:1,000 to 1:100,000
- Process map regions in real-time
- Support 20+ common terrain classification types

---

## Module 5: MapRegistration

### Purpose
Align, georeference, and correct geometric distortions in scanned or photographed maps.

### Agent Generation Prompt
```
Create a MapRegistration module that aligns and georeferences scanned topographic maps, correcting for rotation, scaling, perspective distortion, and coordinate misalignment. The module should:

1. Detect and match coordinate grid references for automatic georeferencing
2. Correct perspective distortion from photographed maps
3. Handle rotation and scaling alignment with reference coordinate systems
4. Register multiple map sheets for seamless tiling
5. Validate registration accuracy using known geographic features

Input: Raw scanned/photographed map images with optional reference coordinates
Output: Geometrically corrected and georeferenced map images

Requirements:
- Automatic registration using grid references and geographic features
- Handle common map projections (UTM, State Plane, etc.)
- Perspective correction for mobile phone captures
- Sub-pixel registration accuracy
- Batch processing for multiple map sheets

Include quality metrics for registration accuracy and coordinate validation.
```

### Expected Interface
```python
class MapRegistration:
    def __init__(self, target_projection: str = "utm", reference_datum: str = "nad83"):
        pass
    
    def auto_register(self, map_image: torch.Tensor, 
                     reference_points: Optional[List[ControlPoint]] = None) -> RegistrationResult:
        """Automatically register map using grid references"""
        pass
    
    def correct_perspective(self, map_image: torch.Tensor) -> torch.Tensor:
        """Correct perspective distortion from camera captures"""
        pass
    
    def align_map_sheets(self, map_images: List[torch.Tensor]) -> torch.Tensor:
        """Align and tile multiple map sheets"""
        pass
    
    def validate_registration(self, registered_map: torch.Tensor, 
                            known_features: List[GeographicFeature]) -> float:
        """Validate registration accuracy against known features"""
        pass

@dataclass
class ControlPoint:
    pixel_coords: Tuple[int, int]
    geo_coords: Tuple[float, float]   # Lat/lon or projected coordinates
    confidence: float

@dataclass
class RegistrationResult:
    registered_image: torch.Tensor
    transform_matrix: torch.Tensor    # 3x3 homography matrix
    registration_error: float        # RMS error in pixels
    geo_bounds: BoundingBox          # Geographic extent
    pixel_scale: float               # Meters per pixel
```

### Validation Criteria
- <10 pixel RMS error for grid-referenced maps
- Handle 0-45° perspective distortion
- Process 4K images in <5 seconds
- Support common map projections and datums

---

## Module 6: FeatureExtractor

### Purpose
Extract geographic features relevant for terrain synthesis trajectory routing and parameter control.

### Agent Generation Prompt
```
Create a FeatureExtractor module that identifies and extracts geographic features from topographic maps for terrain synthesis applications. The module should:

1. Detect ridge lines, valley floors, and watershed boundaries
2. Extract stream and river networks for trajectory routing
3. Identify optimal terrain traversal paths and natural corridors
4. Calculate slope gradients and aspect for terrain characterization
5. Find peaks, saddles, and other prominent topographic features

Input: Heightmap data and classified map features
Output: Structured geographic features suitable for audio synthesis trajectory control

Requirements:
- Robust feature detection across different terrain types
- Real-time processing for interactive trajectory generation
- Configurable feature sensitivity and filtering
- Integration with existing terrain synthesis parameters
- Quality scoring for extracted features

Include spatial analysis tools for feature relationships and connectivity.
```

### Expected Interface
```python
class FeatureExtractor:
    def __init__(self, feature_sensitivity: float = 0.7, min_feature_length: int = 100):
        pass
    
    def extract_ridgelines(self, heightmap: torch.Tensor) -> List[Ridge]:
        """Extract ridge lines for trajectory routing"""
        pass
    
    def find_valleys(self, heightmap: torch.Tensor) -> List[Valley]:
        """Find valley floors and drainage patterns"""
        pass
    
    def detect_peaks(self, heightmap: torch.Tensor) -> List[Peak]:
        """Identify prominent peaks and summits"""
        pass
    
    def calculate_watersheds(self, heightmap: torch.Tensor) -> WatershedData:
        """Calculate watershed boundaries and drainage basins"""
        pass
    
    def find_traversal_paths(self, heightmap: torch.Tensor, 
                           start: Tuple[int, int], 
                           end: Tuple[int, int]) -> TraversalPath:
        """Find optimal terrain traversal between points"""
        pass

@dataclass
class Ridge:
    centerline: torch.Tensor          # (N,2) ridge centerline coordinates
    prominence: float                 # Ridge prominence measure
    elevation_profile: torch.Tensor   # Elevation along ridge
    quality_score: float

@dataclass
class Valley:
    centerline: torch.Tensor          # (N,2) valley floor coordinates
    width_profile: torch.Tensor       # Valley width along centerline
    drainage_area: float              # Upstream drainage area
    gradient: float                   # Average valley gradient

@dataclass
class Peak:
    position: Tuple[int, int]         # Peak location
    elevation: float                  # Peak elevation
    prominence: float                 # Topographic prominence
    isolation: float                  # Distance to higher terrain

@dataclass
class TraversalPath:
    path_coordinates: torch.Tensor    # (N,2) path coordinates
    elevation_profile: torch.Tensor   # Elevation along path
    difficulty_score: float           # Terrain difficulty metric
    total_distance: float
```

### Validation Criteria
- Accurate ridge detection on known mountain ranges
- Find 95%+ of prominent peaks >100m prominence
- Generate natural-looking traversal paths
- Real-time feature extraction for 512x512 heightmaps

---

## Module Integration Strategy

### 1. Processing Pipeline
```
Raw Map Image → MapRegistration → ContourDetector + ElevationOCR + SymbolClassifier
                      ↓
Classified Features → GeometricProcessor → Heightmap + Feature Data
                      ↓
Geographic Features → FeatureExtractor → Trajectory Routes + Terrain Parameters
```

### 2. Agent Generation Sequence
1. **Start with ContourDetector** - foundational for all other modules
2. **Add ElevationOCR** - provides numerical context for contour interpretation
3. **Implement GeometricProcessor** - converts contours to usable heightmaps
4. **Develop SymbolClassifier** - adds semantic understanding of map features
5. **Create MapRegistration** - enables multi-map workflows
6. **Finish with FeatureExtractor** - provides advanced geographic analysis

### 3. Testing and Validation
- Use USGS topographic maps as primary test dataset
- Include international map standards for broader compatibility
- Validate against known geographic databases (DEM, GIS data)
- Performance testing on various hardware configurations

### 4. Integration with Terrain Synthesis
- Heightmaps replace mathematical terrain functions
- Ridge/valley features guide trajectory routing
- Terrain classification influences synthesis parameters
- Real-time updates for interactive geographic exploration

## Success Metrics for Complete Module Set
- Process complete topographic map in <30 seconds
- Generate terrain synthesis data accurate to source map
- Enable real-time geographic terrain exploration
- Support both professional cartographic applications and creative musical use