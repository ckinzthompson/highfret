# highFRET

`highfret` is a Python package for analyzing smFRET data. You can use highFRET from:

1. The graphical interface (GUI)
2. The terminal with the command line interface (CLI)
3. A script

## Read the paper

Gentry, R.C., Leon Hernandez, K.M., Gonzalez, Jr., R.L., Kinz-Thompson, C.D. (202X) *Single-molecule fluorescence imaging at micromolar concentrations without nanophotonic help*. In preparation.

## Installation

There are two ways to install highFRET. If you aren't going to program with highFRET, then you should install the standalone GUI.

#### Install the GUI

(*Note: currently only built for Mac silicon. Windows will come in the future*)

1. Go to the [latest binary release on Github](https://github.com/ckinzthompson/highfret/releases/latest).
2. Download
    1. The `.app` file (Mac).
    2. The `.exe` file (Windows).
3. Double click on the file to launch the GUI.

#### For Scripting/Terminal Use

```bash
pip install git+https://github.com/ckinzthompson/highfret.git
```



# Guide

#### GUI

Watch these videos:
1. [Align the color channels, find spots, then extract traces](https://github.com/ckinzthompson/highfret/blob/cb11d2a98dd3100f265e34019a62cad93e825d55/docs/movies/movie_processing.mp4)
2. [Load traces into tMAVEN](https://github.com/ckinzthompson/highfret/blob/cb11d2a98dd3100f265e34019a62cad93e825d55/docs/movies/movie_analysis.mp4)

#### CLI

After installing the `highfret` python package, you will have access to the CLI program `highfret`. Follow along with these video:

1. [Align the color channels, find spots, then extract traces](https://github.com/ckinzthompson/highfret/blob/cb11d2a98dd3100f265e34019a62cad93e825d55/docs/movies/movie_cli.mp4)
2. If you want to [copy the alignment from one analysis to another you can use the copy function](https://github.com/ckinzthompson/highfret/blob/cb11d2a98dd3100f265e34019a62cad93e825d55/docs/movies/movie_cli_copy.mp4)

#### Stage 1 - Alignment

Polynomial transform of the image estimated by shape analysis
![alignmentimg](https://github.com/user-attachments/assets/07b3db91-1e21-4654-9667-f752a74618c4)

#### Stage 2 - Spotfinding

Image used for spotfinding
![spotfinding 1](https://github.com/user-attachments/assets/d2e0a28f-e730-460a-a50d-ac91a275d5e0)

Spots in each color channel
![spotfinding 2](https://github.com/user-attachments/assets/88ff0b4e-1cbd-4ee8-a5e6-74acdfad2d66)

Final set of spots
![spotfinding 3](https://github.com/user-attachments/assets/8b79ee6f-f1f5-4640-985c-fb41d0b85cb3)

#### Stage 3 - Extraction

Average intensity
![intensity_avg_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/fb95a88d-431b-4dc7-b42b-76a592c65aca)

Example trace
![image2](https://github.com/ckinzthompson/highfret/assets/17210418/ecaa5eb3-59c6-4da2-8850-b0c86f10e2d6)
