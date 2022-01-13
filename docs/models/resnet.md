# ResNet

## General

This model is using a **ResNet** to extract features from the raw polygon's coordinates.

!!! info
    It uses the **official Pytorch** implementation of ResNet18.

## Tokenization

The polygon's coordinates are converted into raster image of black & white, and normalized to 0 ~ 1.

## Feature extraction

Tokens are then embedded, and fed into the ResNet18.  
To match out feature size, additional single layer of dense network is added on the tail of the ResNet.