# Project 2 Machine Learning: Road map segmentation

## Instruction about the framework used in this project
You can create the same conda environment I used during this project. You just need to type: `conda env create -f environment.yml`. The frameworks used in this project are PyTorch (all deep learning part), Torchvision (load dataset + data augmentation), OpenCV (to load or save images), sklearn (f1_score) and Matplotlib (to print images).

## Information about the folders and files
- The folder `dataset` contains the dataset (satellite images with their corresponding groundtruth).
- The folder `models` contains the pretrain model that generated the submission.csv file
- `augmented_dataset.py` is the file that define the dataset to be use with dataloader. Because I do the data augmentation on the fly, all these data augmentations techniques are define in this file.
- `mask_to_submission.py` and `submission_to_mask.py` are the two given files to help for generate submission.csv files.
- `network2.py` is the file that define the Unet network.
- `train2.py` is the file containing the function train (to train the network), accuracy and f1score (to test the score on the validation test).
- `util.py` contains useful functions that help to impement the other files.
- `run.py`is the file that allow to reproduce the same results that I had. More details on it on the next section.
- `Example.ipynb` show an example how to use the different tools I implemented.


## Reproduce my submission.csv file
Just use `python run.py --pretrain` This script will use my pretrain model on the dataset contained in the `dataset folder`, then executed the pretrained model on the test set, generate the results (groundtruth of the images of the test set) and from these results will create the `submission.csv` file.
If you want train from scratch and reproduce my results, you can execute `python run.py --train --batch_size 20 --epoch 500`. In my case, I used a batch size of 20 but you can use a smaller or bigger, it depends on your GPU memory. This command will train my model during 500 epochs with a batch size of 20. It will use the 100 training images and augment them on the fly by applying some random cropping, horizontal or vertical flip and some rotations [-90, -45, 0, 45, 90]. This command will also save the model in `models/new_model`. If you want more information about the model and preprocessing of the images, please go see my Report.pdf.
