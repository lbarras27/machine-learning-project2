# Project 2 Machine Learning: Road map segmentation

## Information about the folders and files
- The folder `dataset` contains the dataset (satellite images with their corresponding groundtruth).
- The folder `models` contains the pretrain model that generated the submission.csv file
- `augmented_dataset.py` is the file that define the dataset to be use with dataloader. Because I do the data augmentation on the fly, all these data augmentations techniques are define in this file.
- `mask_to_submission.py` and `submission_to_mask.py` are the two given files to help for generate submission.csv files.
- `network2.py` is the file that define the Unet network.
- `train2.py` is the file containing the function train (to train the network), accuracy and f1score (to test the score on the validation test).
- `util.py` contains useful functions that help to impement the other files.
- `run.py`is the file that allow to reproduce the same results that I had. More details on it on the next section.
