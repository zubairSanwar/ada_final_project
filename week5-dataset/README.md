# Datasets + Preprocessing

## Dataset Options

I have identified two Xray datasets that I can use for this project:
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?select=chest_xray
    - This dataset has Xrays for Pneumonia classification
    - has 5856 images in total which is a large enough dataset to train a model on
    - it is unclear but I believe the dataset was sourced from multiple sources which means that the xrays could be produced through different protocols which might not be the best for my project as I can't isolate the effect of different protocols on the model's performance
    - Still could potentially use the dataset as it is labelled, large and I could implement transformations to augment 'different protocs'

- https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
    - large 200K+ dataset of xrays for 14 different diseases
    - very large dataset and all sourced from Stanford's hospital so I can assume that the xrays are from the same source; however the time frame for them being taken is very long and could be possible for a protocol or machine change in between
    - the problem with the dataset is that it is way too large to download locally and potentially train a model on
    - another problem is, each xray does not have all 14 labels; so I will have to group the data based on the labels they have to train the model

- https://www.kaggle.com/datasets/nih-chest-xrays/data?select=ARXIV_V5_CHESTXRAY.pdf
    - NIH dataset with about 120K images
    - Each image has a disease label which can be used for classification
    - The dataset however is sourced from several sources which means that the xrays could be produced through different protocols

## Preprocessing steps

Based on my preliminary reading, there are difference in xray imaging based on the machine used or protocol. The differences can be:
- spatial: resolution, field of view, etc
- contrast: contrast settings, use of contrast agents, etc
- noise: different machines have different noise levels, and different protocols can lead to different noise levels

These have to do with the physics of how the xray is taken and the machine used. Since I don't really have access to the machine, the changes will have to be augmented and may not be fully accurate representation of changes in protocols. But I think it will be good enough for the purpose of identifying whether a model trained on one set of medical images can perform well on another set of similar structures but different resolution and contrast. 

To augment the data, the transformations I will apply include: 
- adjusting contrast
- adding noise
- adjusting resolution (e.g. downsampling)
- sharpening or blurring the image

## Self Reflection

I am happy to have found datasets but I am a bit concerned about the fact that the datasets are sourced from multiple sources which means that the xrays could be produced through different protocols which might not be the best for my project as I can't isolate the effect of different protocols on the model's performance. However, I think I can still use the datasets as they are labelled, large and I could implement transformations to augment 'different protocs'. I also feel like I understand the differences in images based on protocols; however a limitation I see is that (since i'm not a radiologist) I might not be able to look at resultant images after transformation and know if they are realistic or not. For this case I also think using CT scans would be a better idea; one difference in protocol for CT scans leads to less number of slices which is a very clear difference that I can easily replicate and evaluate the model on. However, it being 3D would make it difficult to process given that i will train my model locally. I hope my approach seems appropriate to move forward with.