# Multi-Instance-Leaf-Segmentation
Multi-instance leaf segmentation in arabidopsis and tobacco plant images using neural network and image processing techniques
This is a MATLAB R2017a implementation of the watershed segmentation (Wageningen) approach to leaf segmentation in the MVA2016 paper: Leaf segmentation in plant phenotyping: a collation study [1]. 

## Getting Started
The dataset for this work is publicly available [here](http://www.plant-phenotyping.org/datasets). As there are 3 data subsets with different properties, the algorithm has 3 slightly different variations. A data subset is specified in line 5:
```
Training(1); % options {1, 2, 3}
```
The full algorithm can then be run by:
```
Test_Run.m
```
This will save intermediate and final results and show segmentation dice scores and a leaf counting score. Example results of one image per data subset:

<img src="https://github.com/llockhar/Multi-Instance-Leaf-Segmentation/demoImages/SampleResults.png" alt="Leaf Segmentation Results", width="500" />

## Background
Plant testing is performed extensively on model organisms like arabidopsis and tobacco plants to test impacts of changing environmental conditions. Monitoring plant development for dozens or hundreds of plants is time consuming, tedious, and prone to error. Computer vision techniques can enable greater efficiency during plant testing by removing the laborious tasks of counting plant leaves and measuring properties like leaf size. 

Multi-instance leaf segmentation is difficult due to occlusion and similarity in appearance of leaves to one another. Despite these challenges, a technique was developed at Wageningen University to segmentation plant leaves by first separating plant from background using a 2-layer neural network and colourspace transformations and morphological operations. Next, individual leaves were segmented with watershed segmentation and region merging.

<img src="https://github.com/llockhar/Multi-Instance-Leaf-Segmentation/demoImages/WatershedApproach.png" alt="Watershed Approach", width="500" />

[1] Scharr, H., Minervini, M., French, A.P. *et al*. Leaf segmentation in plant phenotyping: a collation study. *Machine Vision and Applications* **27**, 585–606 (2016). https://doi.org/10.1007/s00138-015-0737-3
