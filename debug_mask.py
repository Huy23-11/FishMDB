import sys
sys.path.insert(0, r"C:\Users\messi2015\FishMDB")
from extract_shape_features import ShapeFeatureExtractor

ext = ShapeFeatureExtractor(r"C:\Users\messi2015\FishMDB\FishDataset\processed_u2net\flipped_0131f276b0ecc68f40290e596bb8cc03_nobg.png")
features = ext.extract_all()
print(features is not None)
print(features['fin_ratios'] if features else "FAILED")