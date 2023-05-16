# Delivery Time Prediction for Food apps


## create a conda  virtual environmnent

```
conda  create -p venv python==3.10

conda activate venv/

```
## Installing necessary libraries in python

```
pip install -r requirements.txt

```
## DVC commands

```
dvc init
dvc add src/pipeline/Artifacts/model.pkl
git add src/pipeline/Artifacts/model.pkl.dvc
git add src/pipeline/Artifacts/.gitignore
dvc remote add myremote gdrive://root/DVC_store
dvc push -r myremote

```

## Stack Used

- Python 
- Flask
- Numpy
- Pandas
- sklearn
- Machine Learning
- Random_Forest