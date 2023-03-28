# LT2222 V23 Assignment 3

## General

I did not have time to finish all of the parts, but I have worked on parts 1,2,3,4. I ran out of time towards the end and did not have time to properly document my design choices. I have some documentation throughout my code, that I hope will be enough.

In general the model seems to perform okay-ish, but not great. It reaches an accuracy of 50-something percent, making half of its predictions true. The random partitioning of the data into train and test set might make it so that a class is not present in the training set, meaning the model will never be able to properly predict this label.

I tried to shave off the email headers as best as I could, but sometimes some remnants of them could be left.

## Running a3_features.py

The option available for this file are

+ `inputdir` : The root of the author directories
+ `outputfile` : The name of the output file containing the table of instances
+ `dims` : The output feature dimensions
+ `--test` : The percentage (integer) of instances to label as test

example usages from the root of the project (assuming data/enron_sample is placed in the root as well):

Process the input by reading them from data/enron_sample, keeping 300 feature dimensions and writing the complete output to the file output
`python3 a3_features.py data/enron_sample output 300`

Process the input by reading them from data_enron_sample, keeping 250 feature dimensions and writing the complete output to the file output. The division between training size and test size will be 75/25.
`python3 lt2222-v23-a3/a3_features.py data/enron_sample output 250 --test 25`

## Running a3_model.py

Executing from the root of the project:
`python3 a3_model.py`

There are a couple of options that can configure the execution. I list them here, but they are also documented in the parser-part of the file.

+  `featurefile` : The file containing the table of instances and features
+ `--batchsize` : Batch size to use during training
+ `--epochs` : number of epochs to train for
+ `--learning-rate` : learning rate to use for computing gradients

These three arguments are assumed to be all present, or none of them. Due to lack of time I did not implement error handling for when just a few of them are given.

+ `--use-hidden-with-size` : use a hidden linear layer with a specific size
+ `--act-after-input-layer` : activation function to use after input layer (sigmoid or relu)
+ `--act-after-hidden-layer` : activation function to use after hidden layer (sigmoid or relu)

example usages:

train the model on the contents in output with a batchsize of 30 for 100 epochs
`python3 a3_model.py ./output --batchsize 30 --epochs 100`

train the model similarly as above, but use a hidden layer between the input layer and output layer. After the input layer a rectified linear unit activation function will be applied, and after the hidden layer a sigmoid activation function will be applied.
`python3 a3_model.py ./output --batchsize 30 --epochs 100 --use-hidden-with-size 200 --act-after-input-layer relu --act-after-hidden-layer sigmoid`

## Enron data ethics

I think that one cannot assume privacy when using a company email, but still the distribution of these emails has been much more extensive than I think employees or customers of Enron should have had reason to assume while writing their emails from/to company email addresses. I can understand if the employees and customers feel uncomfortable with this therefore.

I think using the emails in assignments like this one could feel fine to me, since we do not focus on the specific content of the emails or looking up private information that might be in there. I think I would feel differently if we were to use the data to make a searchable website where anyone that knows how to use the internet can search and quickly find "juicy" information about employees or customers mentioned in these emails.