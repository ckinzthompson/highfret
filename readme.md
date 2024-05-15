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

#### Spotfind
Use this to locate donor and acceptor labeled molecules. This acts on the ACF(t=1) image of the movie to find pixel regions not associated with noise.

```python
import highfret
highfret.gui.spotfinder()
```

Data is saved in a folder located where your movie is found. It is titled `/path/to/data/spotfinder_results_<filename>`. Spot files are `<moviename>_<color space>_spots_<origin of spots>.npy`. 

#### Extract
Use this to calculate intensity versus time trajectories from spots located during spotfinding. 

```python
import highfret
highfret.gui.extracter()
```

Data is saved into the same folder as spotfinding data. Intensities are saved as an HDF5 file (under group `data`) called `intensities_<moviename>.hdf5`. They can be open in [tMAVEN](https://gonzalezbiophysicslab.github.io/tmaven/). Use `File` > `Load` > `HDF5 Dataset` > `Raw`, then click the `data` group name in the popup and click the `Select` button.

