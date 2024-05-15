# highFRET
`highfret` is a Python package for analyzing smFRET data


## Directions

### Installation
```bash
conda create -n highfret python
conda activate highfret
pip install highfret
```

### Use
The `highfret` GUI is made to run in a Jupyter Notebook. Ideally, you can save these notebooks in the folders where your data live. First, you will need to launch Jupyter Labs:

```bash
conda activate highfret
jupyter-lab
```

Run the following stages (probably in order)

#### Alignment 
Use this to map the donor (green) and acceptor (red) color channels into each other.

```python
import highfret
highfret.gui.aligner()
```

Data is saved in a folder located where your movie is found. It is titled `/path/to/data/aligner_results_<filename>`. Alignment files are polynomial tranform coefficients that map acceptor (red) into donor (green) space.

<img width="1391" alt="alignment gui" src="https://github.com/ckinzthompson/highfret/assets/17210418/81478da4-ffa5-4c81-9724-8c9741c271c4">

#### Spotfind
Use this to locate donor and acceptor labeled molecules. This acts on the ACF(t=1) image of the movie to find pixel regions not associated with noise.

```python
import highfret
highfret.gui.spotfinder()
```

Data is saved in a folder located where your movie is found. It is titled `/path/to/data/spotfinder_results_<filename>`. Spot files are `<moviename>_<color space>_spots_<origin of spots>.npy`. 

<img width="1383" alt="spotfind gui" src="https://github.com/ckinzthompson/highfret/assets/17210418/5d2d4618-c2ae-4d5b-b3a6-0f1805271366">

#### Extract
Use this to calculate intensity versus time trajectories from spots located during spotfinding. 

```python
import highfret
highfret.gui.extracter()
```

Data is saved into the same folder as spotfinding data. Intensities are saved as an HDF5 file (under group `data`) called `intensities_<moviename>.hdf5`. They can be open in [tMAVEN](https://gonzalezbiophysicslab.github.io/tmaven/). Use `File` > `Load` > `HDF5 Dataset` > `Raw`, then click the `data` group name in the popup and click the `Select` button.

<img width="1381" alt="extract gui" src="https://github.com/ckinzthompson/highfret/assets/17210418/82436f80-8521-4051-9563-d567df0a351d">
