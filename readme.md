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

Follow along with these videos:

Align the color channels, find spots, then extract traces

![prepare](https://github.com/user-attachments/assets/d4bbfeec-5076-4dec-b915-dd3d50fdb8e1)

Load traces into [tMAVEN](https://gonzalezbiophysicslab.github.io/tmaven/)

![anlysis](https://github.com/user-attachments/assets/47d8084c-fbe3-4370-9bd7-2816c97143e0)

#### CLI

After installing the `highfret` python package, you will have access to the CLI program `highfret`. Follow along with this video:

![cli](https://github.com/user-attachments/assets/d4b24803-42a6-44f8-beb7-47c14eadd3d8)

If you want to copy the alignment from one analysis to another you can use the copy function

![cli_copy](https://github.com/user-attachments/assets/8537a246-a782-49ba-b50b-2b93b2eecc6c)

or for example just `cp` from movie 5 to movie 4:
`cp highfret_5/transforms.npy highfret_4/tranforms.npy`

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
