import ee

# --- Authentication using your Service Account ---
# Make sure you have your credentials.json file and update the path.
SERVICE_ACCOUNT = 'earth-engine-api@leafy-emblem-474119-k7.iam.gserviceaccount.com'
KEY_FILE = 'credentials.json' # <-- IMPORTANT: Update this path

# Initialize the library with your credentials.
try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
    ee.Initialize(credentials=credentials)
except Exception as e:
    print(f"Authentication failed: {e}")
    print("Please ensure your SERVICE_ACCOUNT and KEY_FILE path are correct.")
    exit()

# --- Define Area of Interest (AOI) ---
# Using the same sample coordinates in Mumbai.
user_aoi = {
  "type": "Polygon",
  "coordinates": [
    [
      [72.48, 18.53],
      [72.57, 19.01],
      [72.58, 19.10],
      [72.53, 19.16],
      [72.47, 19.16],
      [72.46, 19.10],
      [72.49, 19.06]
    ]
  ]
}
aoi = ee.Geometry.Polygon(user_aoi['coordinates'])

# --- Get a single, cloud-free satellite image ---
# This composite image will be the source for all our calculations.
image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
         .filterBounds(aoi)
         .filterDate('2024-10-01', '2025-01-01')
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
         .median())

# --- Calculate all three indices ---

# 1. Normalized Difference Water Index (NDWI)
# Formula: (Green - NIR) / (Green + NIR)
ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

# 2. Normalized Difference Built-up Index (NDBI)
# Formula: (SWIR1 - NIR) / (SWIR1 + NIR)
ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

# 3. Soil-Adjusted Vegetation Index (SAVI)
# Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L), where L=0.5
savi = image.expression(
    '1.5 * (NIR - RED) / (NIR + RED + 0.5)', {
        'NIR': image.select('B8'),
        'RED': image.select('B4')
    }).rename('SAVI')

# --- Combine all index bands into one image ---
# This is more efficient than calculating statistics for each one separately.
final_image = ndwi.addBands(ndbi).addBands(savi)

# --- Calculate the average value for all indices within the AOI ---
# We use a single reduceRegion call with a mean() reducer.
stats = final_image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=aoi,
    scale=30
)

# --- Get the results and print them ---
results = stats.getInfo()

print("--- Index Results for the Selected Area ---")
# The '.get' method is used to safely access the dictionary keys
print(f"The average NDVI for the selected area is: {result.get('NDVI', 'N/A'):.4f}")
print(f"💧 Average Normalized Difference Water Index (NDWI): {results.get('NDWI', 'N/A'):.4f}")
print(f"🏙️  Average Normalized Difference Built-up Index (NDBI): {results.get('NDBI', 'N/A'):.4f}")
print(f"🌱 Average Soil-Adjusted Vegetation Index (SAVI): {results.get('SAVI', 'N/A'):.4f}")