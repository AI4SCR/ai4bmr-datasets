# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install git+https://github.com/AI4SCR/ai4bmr-datasets.git

rm -r /work/FAC/FBM/DBC/mrapsoma/prometex/issues/ai4bmr-datasets/25/Keren2018
python /work/FAC/FBM/DBC/mrapsoma/prometex/projects/ai4bmr-datasets/issues/issue-25.py