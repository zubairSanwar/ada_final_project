# Reflection

In this .md i have my reflection and also some info on the different files/folders

## Directory and Files

- `models`
    - this holds the saved weights, biases, etc from the trained models so I could run inferences and test the model
    - I couldn't push the models as the `.keras` files were too large
- `training_curves`
    - since I used the same code to train both models, I saved images of their learning and loss curves if you want to check them
- `parse_dataset.py`
    - I used this script to walk through the dataset, and save images paths as a csv with its labels so I can easily import the images and labels while training the model
- `train_CNN.ipynb`
    - This holds the code for designing and training the model, as well as preparing the dataset for training. The same code was used to train model on original and transformed dataset
- `testing_models.ipynb`
    - This is where I uploaded the models and tested their performance on both the original and tranformed dataset

## What I achieved

I chose to use the first dataset in my project with xrays and labels of Pneuomonia/healthy. First of all, I updated my image processing code from week5 to go through the entire xray dataset, and transform it. I adjusted the contrast, blur, noise, and other visual toggles to augment xrays taken from a different machine or protocol. Since I am not a radiologist, I did use Claude to figure out what values I could use in my transformations which would be realistic to see in xrays from different machinery/protocols. 

After processing the whole dataset I now had the original xrays and the augmented transformed xrays mimicing images obtained from a differnet protocol. It's important to note that the underlying image is the same, just the visual features (like resolution, etc) is adjusted; so the 'signs' for pneumonia stay the same. After doing this, I trained 2 models; the model design stayed the same but one was trained on the original xray dataset and one was trained on the transformed xray dataset. I trained both locally on my macbook. This was probably the most challenging part.

After training the model, I tested both the models on the dataset it was trained on and the other dataset to see how performance changes based on the quality of the dataset. You can see my analysis in the `testing_models.ipynb` notebook; I did a quick short analysis at the end and aim to go deeper in the final checkpoint. 

There are clear signs of model dependence on quality of medical information/protocol followed to obtain the medical images. 

## What am I happy about

I am proud of the progress I made, which is a lot compared to week 5. I ran into a lot of issues in the model training, however I fixed it so it trains faster locally, the model learns properly, etc. I am also proud of the decisions I made as I think it shows my understanding; this includes training a simple CNN as the task is very simple and so I think a denser model would lead to overfitting. I am also happy to see results I expected and ones I have somewhat seen in past research work; I am proud to be able to validate this understanding.

## Struggles

In this checkpoint, training the model was hard. First of all I was training it locally which was taking a long time as first i was doing it on the CPU. On looking up online, I learned how to run it on the mac gpu which brought traiing time down to 20-25 mintues compared to 1+ hour. Another problem was that the model initially was not learning well, and i realised it was due to the class imbalance in my dataset. To be honest, I used claude to understand how to fix that which is how I added the class weight code in my implementation. 

For future, I think i need to do better fine grained analysis of the results. I also feel like my model is way too simple, so i don't know if i should also test using different models like ViT but I can't train that locally. I could use a pretrained ViT and fine tune it, but that would be a lot more complex to not fall for the different imaging protocol study. However, I still want to put a pin in that and see if i can do that for the final to show how pretrained models are not a bad idea to use for these kind of tasks as they have a good understanding of structures, etc before finetuning on a given task. 

## Specific Focus for course staff

My next step is to train a model on the a combination of original and transformed xrays and test its performance and then do deeper analysis on whether that's a good solution and elaborate more on model perforamance on different xray qualities. I want to know if my current analysis has any pitfalls, if my future direction is a good idea, and if fine tuning a ViT is doable for this project and something I should look into?