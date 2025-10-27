/**
 * Land Cover Classification - [KHDTK_UGM_GETAS]
 * @author Hilmi Nur Persada
 * @description Script untuk klasifikasi tutupan lahan menggunakan machine learning
 * @created [2022]
 * @version 1.0
 */// Load Koleksi Citra Landsat 8 Collection 2 Tier 1 Realtime TOA
var landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_RT_TOA")
  .filterBounds(table) // Filter berdasarkan AOI
  .filterDate('2022-01-01', '2022-12-31') // Tetap menggunakan tahun 2006 (CATATAN: Landsat 8 diluncurkan 2013, tanggal perlu disesuaikan!)
  .filter(ee.Filter.lt('CLOUD_COVER', 15)) // Persentase awan < 10%
  .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']); // Band utama Landsat 8 (Blue, Green, Red, NIR, SWIR1, SWIR2)

// Hitung NDVI untuk setiap citra dalam koleksi
var ndviCollection = landsat8.map(function(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI'); // B5=NIR, B4=Red untuk Landsat 8
  return image.addBands(ndvi);
});

// Ambil citra median untuk mengurangi noise
var all_bands = ndviCollection.median().clip(table);
print('All bands', all_bands);

// Analisis statistik nilai min dan max band (opsional, untuk penyesuaian visualisasi)
var statsB4 = all_bands.select('B4').reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: table,
  scale: 30,
  maxPixels: 1e9
});
print('Statistik B4 (Red):', statsB4);


// Tampilkan Citra di Peta (RGB = B4, B3, B2 untuk Landsat 8)
Map.centerObject(table, 10);
Map.addLayer(
  all_bands,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3, gamma: 1.4}, // Sesuaikan min/max sesuai kebutuhan
  'Landsat 8 RGB'
);

// Tambahkan layer NDVI
var ndviParams = {
  min: -1,
  max: 1,
  palette: ['blue', 'white', 'green']
};
Map.addLayer(all_bands.select('NDVI'), ndviParams, 'NDVI 2006');

// Tambahkan layer false color (NIR, Red, Green) untuk analisis vegetasi
Map.addLayer(
  all_bands,
  {bands: ['B5', 'B4', 'B3'], min: 0, max: 0.5, gamma: 1.4}, // B5=NIR, B4=Red, B3=Green
  'False Color 2006 (NIR-R-G)'
);

// Ekspor hasil
Export.image.toDrive({
  image: all_bands.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI']), // Band Landsat 8
  description: 'Landsat8',
  folder: 'SKRIPSI',
  fileNamePrefix: 'L8_2006',
  region: table,
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

var all_bands = all_bands;

var Hutan_Jati = Hutan_Jati.map(function(f) {
  return f.set('kelas', 1);
});

var Perkebunan_Tebu = Perkebunan_Tebu.map(function(f) {
  return f.set('kelas', 2);
});

var Ladang_Jagung = Ladang_Jagung.map(function(f) {
  return f.set('kelas', 3);
});

var Semak_Belukar = Semak_Belukar.map(function(f) {
  return f.set('kelas', 4);
});

var Sawah = Sawah.map(function(f) {
  return f.set('kelas', 5);
});

var Lahan_Terbuka = Lahan_Terbuka.map(function(f) {
  return f.set('kelas', 6);
});

// Gabungkan semua kelas menjadi satu FeatureCollection
var kelas = Hutan_Jati
  .merge(Perkebunan_Tebu)
  .merge(Ladang_Jagung)
  .merge(Semak_Belukar)
  .merge(Sawah)
  .merge(Lahan_Terbuka)
var bands = ['B5', 'B2', 'B3', 'B4', 'NDVI'];
var selected_input = all_bands.select(bands);

var samples = selected_input.sampleRegions({
  collection: kelas ,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Ht = selected_input.sampleRegions({
  collection: Hutan_Jati,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Pt = selected_input.sampleRegions({
  collection: Perkebunan_Tebu,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Lj = selected_input.sampleRegions({
  collection: Ladang_Jagung,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Sb = selected_input.sampleRegions({
  collection: Semak_Belukar,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Sa = selected_input.sampleRegions({
  collection: Sawah,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

var Tot_Lt = selected_input.sampleRegions({
  collection: Lahan_Terbuka,
  properties: ['lc'],
  scale: 30
}).randomColumn('random');

print('Total Sampel n =', samples.aggregate_count('.all'));
print('Total Sampel Hutan_Jati =', Tot_Ht.aggregate_count('.all'));
print('Total Sampel Perkebunan_Tebu =', Tot_Pt.aggregate_count('.all'));
print('Total Sampel Ladang_Jagung =', Tot_Lj.aggregate_count('.all'));
print('Total Sampel Semak_Belukar =', Tot_Sb.aggregate_count('.all'));
print('Total Sampel Sawah =', Tot_Sa.aggregate_count('.all'));
print('Total Sampel Lahan_Terbuka =', Tot_Lt.aggregate_count('.all'));

// Uji akurasi = 70% (training) dan 30% (validasi/testing)
// Split Sample
var split = 0.7;
var training = samples.filter(ee.Filter.lt('random', split));
var testing = samples.filter(ee.Filter.gte('random', split));
print('Training n =', training.aggregate_count('.all'));
print('Testing n =', testing.aggregate_count('.all'));


// 9. Latih model Random Forest
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  variablesPerSplit: null,
  minLeafPopulation: 1,
  bagFraction: 0.5,
  maxNodes: null,
  seed: 0
}).train({
  features: samples,
  classProperty: 'lc',
  inputProperties: bands
});

// 10. Terapkan klasifikasi ke citra
var classified = all_bands.select(bands).classify(classifier);

var tuplah = selected_input.classify(classifier);
Map.addLayer(tuplah, {min: 1, max: 6, palette: ['167525', '20ff24', 'e1e14f', '02360c','b2ffbe','f63030']}, "SHP");

// Uji Akurasi
var validation = testing.classify(classifier);
var testAccuracy = validation.errorMatrix('lc', 'classification');
print('Validation Error Matrix RF =', testAccuracy);
print('Validation Overall Accuracy RF =', testAccuracy.accuracy());


// Ekspor citra hasil klasifikasi
Export.image.toDrive({
  image: tuplah, // Citra hasil klasifikasi
  description: 'Hasil_Klasifikasi_6_Kelas',
  folder: 'SKRIPSI',
  fileNamePrefix: 'Klasifikasi',
  region: table,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});
