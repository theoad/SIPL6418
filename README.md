# SIPL6418
Investigate no-reference metrics for discontinuous, high dimensional distributions

## Setup
```
conda create --name sipl
conda activate sipl
pip install -r requirements.txt
```

## Usage
### Running the script
```
python main.py
```

### Changing arguments
Modify the following lines in main.py
```python
METRIC = 'fid'  # 'rid', 'fid'
MODEL = 'inception'  # 'inception', 'clip'
CHANNELS = 2048  # 64, 192, 768, 2048
SCORING = 'knn'  # 'svm', 'knn'; ignored if METRIC == 'fid'
```
