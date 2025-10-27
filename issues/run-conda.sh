# Create a conda environment
conda create -n ai4bmr-datasets python=3.12 -y
conda activate ai4bmr-datasets

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git
#conda install conda-forge::r-base  # optional, if R is not already installed

rm -r /work/FAC/FBM/DBC/mrapsoma/prometex/issues/ai4bmr-datasets/25/Keren2018
python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-datasets/issues/issue-25.py