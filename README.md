# ITDS

This repository contains code to reproduce results from the paper:

[Enhancing Transferability of Targeted Adversarial Examples via Inverse Target Gradient Competition and Spatial Distance Stretching]()

## Requirements

```bash
Python >= 3.9.13
torch >= 1.11.0
torchvision >= 0.12.0
numpy >= 1.23.3
scipy >= 1.9.3
```

## Quick Start

### Prepare the data and models

For data, you can place data in `./eval_data/`.  
For models, all normal training models are from the pytorch official website, and the calling method is already in the `craft.py` and `eval.py`, or you can consult the official documentation to use other models. And for all adversarial training defense models, you can download from [here](https://drive.google.com/file/d/13DcsFytr4P1A52xwvbvkg2TS2tL185Oe/view?usp=sharing) and place it in `./defense_models/`, their calling methods are also integrated into our codes (Have been commented).

### Running Attack

Taking ITDS attack for example, you can run this attack as following: `CUDA_VISIBLE_DEVICES=gpuid python craft.py`

### Evaluating the Attack

The generated adversarial examples would be stored in directory `./adv_examples/`. Then run the file ASR_eval.py to evaluate the success rate of each model used in the paper: `CUDA_VISIBLE_DEVICES=gpuid python eval.py`
