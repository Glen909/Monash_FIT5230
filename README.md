# Adversarial Stickers: A Stealthy Attack Method in the Physical World (Modified)

This repository is based on the code from the following paper:  
[Adversarial Stickers: A Stealthy Attack Method in the Physical World](https://ieeexplore.ieee.org/abstract/document/9779913) (TPAMI 2022). We have made modifications to extend or enhance the original implementation. Please refer to the report for detailed explanations of our changes or use code diff to inspect the modifications.

## Preparation

### Environment Settings:

To set up the environment, please install the required dependencies from the provided `requirements.txt` file:


This will ensure all necessary libraries and versions are properly installed. Adjustments to the environment should be made based on your system configuration if necessary.

### Data Preparation:
We follow the same structure for data preparation, with minor modifications to improve dataset handling. These adjustments include **[describe any changes made to dataset preprocessing, handling, etc.]**.

+ Face Data:
Please download the dataset ([LFW](http://vis-www.cs.umass.edu/lfw/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) and place it in `./datasets/`.

The directory structure remains the same:


+ Stickers:
Pre-defined stickers should still be placed in `./stickers/`. **[Mention any changes if you've added or altered sticker generation methods or pre-processing steps.]**

### Model Preparation:
Tool models ([FaceNet](https://github.com/timesler/facenet-pytorch), [CosFace](https://github.com/deepinsight/insightface/tree/master/recognition), [SphereFace](https://github.com/clcarwin/sphereface_pytorch)) are required and should be placed in `./models/`.

Make sure to adjust the corresponding `./utils/predict.py` if necessary. **[Mention any modifications made to the model loading or prediction pipeline.]**

### Other Necessary Tools:
The additional tools and data needed remain the same, but ensure they align with any changes made in the modified version.

+ Python tools for [3D face](https://github.com/YadiraF/face3d/tree/master/face3d)
+ BFM Data: `./BFM/BFM.mat`
+ Shape predictor for face landmarks ([68](https://github.com/r4onlyrishabh/facial-detection/tree/master/dataset), [81](https://github.com/codeniko/shape_predictor_81_face_landmarks))

## Quick Start
The hyperparameter settings are still managed in `./utils/config.py`. Please refer to our report for any modifications to these settings or review the code for specific updates.

To run an attack, use the following command:

Make sure to adjust paths and configurations based on the changes described above.

## Citation
If you find the original methods useful, please consider citing the original paper: