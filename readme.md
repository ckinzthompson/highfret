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

## Guide

#### GUI

Follow along with these videos:

Align the color channels, find spots, then extract traces
![prepare]()

Load traces into [tMAVEN](https://gonzalezbiophysicslab.github.io/tmaven/)
![anlysis]()

#### CLI

After installing the `highfret` python package, you will have access to the CLI program `highfret`. Follow along with this video:

![cli]()

If you want to copy the alignment from one analysis to another you can use the copy function

![cli_copy]()

or for example just `cp` from movie 5 to movie 4:
`cp highfret_5/transforms.npy highfret_4/tranforms.npy`

#### Stage 1 - Alignment

Polynomial transform of the image estimated by shape analysis
![alignmentimg]

#### Stage 2 - Spotfinding

Image used for spotfinding
![spotfinding 1]

Spots in each color channel
![spotfinding 2]

Final set of spots
![spotfinding 3]

#### Stage 3 - Extraction

Average intensity
![intensity_avg_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/fb95a88d-431b-4dc7-b42b-76a592c65aca)

Example trace
![image2](https://github.com/ckinzthompson/highfret/assets/17210418/ecaa5eb3-59c6-4da2-8850-b0c86f10e2d6)