# Folder 'src'
This is directory for source files using in current project.

## Generate dataset

Example of standart usage:
```
python generate_dataset.py -i ../data/27612.01.01.2005.11.08.2017.1.0.0.en.utf8.00000000.csv -o ../data/moscow.dump --interpolate --normalize --feature_names T Po U
```

## Train model

Example of standart usage:
```
python train.py -i ../data/moscow.dump --log_dir /tmp/forecast --epoch_count 1
```