# Dataset

Utilities to data handling in the AI4BMR group.

# Pre-requisites

- Access to the github.com/AI4SCR
- Access to the cluster
- Access to the project `prometex` (maybe you already have read access, just try)

# Installation

```bash
git clone git+ssh://git@github.com/AI4SCR/ai4bmr-core.git
git clone git+ssh://git@github.com/AI4SCR/ai4bmr-datasets.git
pip install -e ai4bmr-core
pip install -e ai4bmr-datasets
```

# Example

You can either run the code below directly on the cluster or you need to download the data first.

## Download data from cluster

If you want to play, start with `TNBC`, this is
a [public dataset](https://www.sciencedirect.com/science/article/pii/S0092867418311000?via%3Dihub)
and it is the smallest of the three available datasets.

### (Optional) Configure access to the UNIL cluster via SSH

- Create an SSH key pair (see online tutorials search for `ssh-keygen` or refer to our internal wiki)
- If you do not have an SSH config file yet create one:

```bash
touch ~/.ssh/config
```

- Add the following configuration to the file

```bash
Host unil
    Hostname curnagl.dcsr.unil.ch
    AddKeysToAgent Yes
    ForwardAgent yes
    IdentityFile ~/.ssh/<NAME_OF_YOUR_PRIVATE_KEY>
    User <USERNAME>
```

### Download the data

- Create a dataset folder

```bash
# create data folder, adapt as needed
mkdir -p ~/data/datasets
cd ~/data/datasets || exit
```

- Download the data with `rsync`
- If you did not configure your `~/.ssh/config` file you need to replace `unil` with `<USERNAME>@curnagl.dcsr.unil.ch`

```bash
# all datasets
rsync -ahvP unil:"/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/*.tar" .
# specific dataset
rsync -ahvP unil:/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/TNBC.tar .
# unzip the dataset
tar -xvf TNBC.tar
```

## Load TNBC data

- This is an example, access for `BLCa` and `PCa` is similar.

```python
from pathlib import Path

from ai4bmr_datasets.datasets import TNBC

# NOTE: on the cluster use: /work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/TNBC
base_dir = Path('~/data/datasets/TNBC').expanduser()
dataset = TNBC(base_dir=base_dir)
dataset.setup()
data = dataset.load()

# data is a dictionary
print(data.keys())

# access images, checkout the [Image.py](src%2Fai4bmr_datasets%2Fdatamodels%2FImage.py)
# the returned images are only loaded into memory upon calling the `.data` attribute
# an image is ~50MB
images = data['images']
sample_id = list(images.keys())[0]
image = images[sample_id]
print(image.data.shape)  # C x H x W
image.metadata  # metadata of the image, aka the panel, also accesible via data['panel']

# segmentation mask of the image
# the returned mask are only loaded into memory upon calling the `.data` attribute
mask = data['masks'][image.id]
mask.data.shape
assert mask.data.shape == image.data.shape[1:]
mask.metadata  # metadata of the mask, not yet implemented for BLCa, PCa

# get single cell data for the image
df = data['data']
sc = df[(df.index.get_level_values('sample_id') == image.id)]

# get patient data for the image
data['patient_data'].loc[image.id]

```
