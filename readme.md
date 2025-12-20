# highFRET
`highfret` is a Python package for analyzing smFRET data
![spots_all_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/25c02463-cfd7-4281-8973-8f6c2d007b08)

## Directions
Follow these instructions.

### Installation
```bash
conda create -n highfret python
conda activate highfret
pip install git+https://github.com/ckinzthompson/highfret.git
```


### Use
![intensity_avg_vc436HS_1_MMStack_Pos0](https://github.com/ckinzthompson/highfret/assets/17210418/fb95a88d-431b-4dc7-b42b-76a592c65aca)

![image2](https://github.com/ckinzthompson/highfret/assets/17210418/ecaa5eb3-59c6-4da2-8850-b0c86f10e2d6)


### Dev notes
`pyinstaller --onedir --icon=docs/graphics/logo.png --windowed --name "highFRET" --add-data "highfret/web/templates:templates" highfret/web/webapp.py`