# COCO 2014 Dataset Setup

Experiment 1 requires the COCO 2014 validation dataset:
1. Validation images directory: `val2014/` (containing images formatted as `COCO_val2014_000000xxxxxx.jpg`)
2. Instances annotations file: `instances_val2014.json`
3. Captions annotations file: `captions_val2014.json`
4. Synonyms file: `synonyms.txt` (provided in this directory)

### Official Data Acquisition:
```bash
# Download annotations:
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
# Move instances_val2014.json and captions_val2014.json into data/

# Download val2014 images (~6.2 GB):
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/
```

### On Kaggle:
The notebook `notebooks/run_kaggle.ipynb` includes automated download cells for annotations and dataset integration.
