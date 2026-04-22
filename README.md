# ADA Final Project

## Analysing Medical Image Model's Dependence on Imaging Protocols.

To elaborate, in my past projects I came to learn different medical facilities have different imaging protocols; it may depend on machine used, contrast setting, etc. Different protocols lead to differences in images in things like resolution, contrast, noise levels. In most research papers, models are trained with data sourced from a single source which makes me question: are these models general enough? Would they start acting up when processing images produced through different protocols? Can such a model perform well generally, or is it better to train with data sourced from different sources?

To answer these questions, I will train a general CNN based model on a dataset of Xrays and compare it's performance on a test set of Xrays that are either sourced from a different dataset or transformed to replicate different protocols. I will also train another model on the transformed images and compare the performance of the two models to see if there's a difference.


### Roadmap for the project:

First steps

- find a dataset of Xrays (2D and therefore easier to process) with a label for classification (e.g. lung cancer or not) to train a model on
- do a literature review to understand the different imaging protocols and how they affect the images, and find out what transformations can be applied to replicate the effects of different protocols
- perform the transformations on the images to replicate different protocols

Second steps

- train a classification CNN on original xray images
- evaluate model on original xray test set and transformed xray test set to see if there's a difference in performance
- train another classification CNN on transformed xray images

Final steps

- analyse performance of both models on the datasets
- possibly if I have time, train a model on a combined dataset of original and transformed images to see if it performs better than the two models trained separately
- report findings and evaluate tradeoffs of training on a combined dataset vs training on separate datasets and whether this is a valid concern (also understand limitations of my approach)


