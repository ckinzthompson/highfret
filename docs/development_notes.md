# Development Notes

## Organization
Users make a container (`highfret/highfret/containers/tif_folder.py`). These keep all the data together. The minimal requirements are given in an abstract base class (`highfret/highfret/containers/general.py`). There might be a slight issue where the CLI (`highfret/highfret/cli.py`) has some specificity for the `tif_folder` container, but it should be easily fixed.

In general, the functions in `highfret/highret` files (e.g., `highfret/highfret/aligner.py`) use the container. The more math-heavy functions that do the actual work, but that don't use containers (e.g., they take something held in a container), live in `highfret/highfret/support`

## Create installer
`pyinstaller --onedir --icon=docs/graphics/logo.png --windowed --name "highFRET" --add-data "highfret/web/templates:templates" highfret/web/webapp.py`