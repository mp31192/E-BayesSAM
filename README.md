# E-BayesSAM
This repository contains the official implementation of MICCAI 2025 accepted paper **E-BayesSAM: Efficient Bayesian Adaptation of SAM with Self-Optimizing KAN-Based Interpretation for Uncertainty-Aware Ultrasonic Segmentation**.

## Model Checkpoint
You can download the model checkpoint in this [**anonymous link**](https://drive.google.com/file/d/14LGJloMNxAzlWp9KcmR01BWoEimUA5wI/view?usp=drive_link).

Please place the downloaded "Causal_SAM_checkpoint.pt" file in the root directory of this project.

## How to run
To perform inference using the model, run the following script:
```
python inference_CausalSAM_with_UncertaintyPrompt.py -p examples/000026.png
```
To visualize the uncertainty map along with the segmentation result, use:
```
python inference_CausalSAM_with_UncertaintyPrompt.py -p examples/000026.png -u True
```

## Example
Below is the input image used for the example:

![image](https://github.com/mp31192/Causal-SAM/blob/main/examples/000026.png)

You can manually obtain the box prompt as shown below:

![image](https://github.com/mp31192/Causal-SAM/blob/main/img/BoxPrompt_Manually.png)
![image](https://github.com/mp31192/Causal-SAM/blob/main/img/BoxPrompt.png)

Here is the segmentation result produced by the model:

![image](https://github.com/mp31192/Causal-SAM/blob/main/examples/000026_result.png)

To also visualize the uncertainty map:

![image](https://github.com/mp31192/Causal-SAM/blob/main/examples/000026_result_ShowUncertainty.png)

## Todo list
- [ ] Release training code

## Acknowledgement
We would like to thank the developers of [**SAMMED2D**](https://github.com/OpenGVLab/SAM-Med2D) and [**MedSAM**](https://github.com/bowang-lab/MedSAM) for their valuable contributions.
