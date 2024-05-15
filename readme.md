# highFRET
`highfret` is a Python package for analyzing smFRET data
![spots_all_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/25c02463-cfd7-4281-8973-8f6c2d007b08)

## Directions
Follow these instructions.

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

Run the following stages (probably in order):

#### Alignment 
Use this to map the donor (green) and acceptor (red) color channels into each other.

```python
import highfret
highfret.gui.aligner()
```

Data is saved in a folder located where your movie is found. It is titled `/path/to/data/aligner_results_<filename>`. Alignment files are polynomial tranform coefficients that map acceptor (red) into donor (green) space.
<img width="1391" alt="alignment gui" src="https://github.com/ckinzthompson/highfret/assets/17210418/81478da4-ffa5-4c81-9724-8c9741c271c4">

Here is the view of the prealigned images (left), aligned images (center), and transformed grid.
![render_0007_optimize_order3_bin1](https://github.com/ckinzthompson/highfret/assets/17210418/4f8ce072-648b-4043-8b3b-f7e8efc39d04)

#### Spotfind
Use this to locate donor and acceptor labeled molecules. This acts on the ACF(t=1) image of the movie to find pixel regions not associated with noise.

```python
import highfret
highfret.gui.spotfinder()
```

Data is saved in a folder located where your movie is found. It is titled `/path/to/data/spotfinder_results_<filename>`. Spot files are `<moviename>_<color space>_spots_<origin of spots>.npy`. 
<img width="1383" alt="spotfind gui" src="https://github.com/ckinzthompson/highfret/assets/17210418/5d2d4618-c2ae-4d5b-b3a6-0f1805271366">

The aligned images used for spotfinding.
![overlay_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/3f999a4d-55a1-4fbc-a2c5-d44c542e9f60)

Here you can see the spots that have been found. Note these are only those that colocalize in both color channels.
![spots_final_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/242875d1-7b1a-43bf-bbfc-8048af370dbe)

#### Extract
Use this to calculate intensity versus time trajectories from spots located during spotfinding. 

```python
import highfret
highfret.gui.extracter()
```

Data is saved into the same folder as spotfinding data. Intensities are saved as an HDF5 file (under group `data`) called `intensities_<moviename>.hdf5`. They can be open in [tMAVEN](https://gonzalezbiophysicslab.github.io/tmaven/). Use `File` > `Load` > `HDF5 Dataset` > `Raw`, then click the `data` group name in the popup and click the `Select` button.

After extracting intensities, a plot of the average intensity versus time is provided so you can assess how it went.
<img width="1381" alt="extract GUI" src="https://github.com/ckinzthompson/highfret/assets/17210418/f45cbe5f-3453-4ebe-b382-55fee2b07a5d">

and FINALLY we have traces!!!!
![intensity_avg_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/fb95a88d-431b-4dc7-b42b-76a592c65aca)

![image2](https://github.com/ckinzthompson/highfret/assets/17210418/ecaa5eb3-59c6-4da2-8850-b0c86f10e2d6)


