# Data

**Dataset:** NEU surface defect database (NEU-DET), hot-rolled steel strip, Northeastern University
(Song Kechen and Yan Yunhui). 200 x 200 grey images, six defect types, one box per defect.

**Source used here:** the cleaned release `KeenForgeAI/NEU-DET-corrected` on Hugging Face, pinned to
commit `cbb799e5e5b8198d6228d94778f835082e42883c`. It is the official data with 3 duplicate images and
3 duplicate boxes removed: 1,797 images and 4,177 boxes.

**The dataset is not stored in this repository.** `python scripts/download_data.py` downloads it to
`data/neu/` and converts the Pascal VOC annotations to YOLO labels.

| Path | In git | What |
|---|---|---|
| `splits/train.txt`, `val.txt`, `test.txt` | yes | The image names of each split (seed 42, 70/15/15, balanced by defect type) |
| `samples/` | yes | Twelve test images, two per defect type, for the demo and the tests |
| `neu/` | no | The downloaded dataset and the converted labels |

## Licence of the images

The original authors publish NEU-DET for research and ask for a citation; they state no licence.
The twelve sample images in `samples/` are included for demonstration only, with attribution.
The MIT licence of this repository covers the code, not these images. Before using the dataset
in a commercial product, ask the authors for their terms, or train on your own images.

## Citation

- K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled
  steel strip surface defects", Applied Surface Science 285, 2013. doi:10.1016/j.apsusc.2013.08.123
- Y. He, K. Song, Q. Meng and Y. Yan, "An end-to-end steel surface defect detection approach via
  fusing multiple hierarchical features", IEEE Trans. Instrumentation and Measurement 69(4), 2020.
  doi:10.1109/TIM.2019.2915404
