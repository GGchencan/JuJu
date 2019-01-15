"""Read, split and save the juju dataset for our model"""

import csv
import os
import sys


def load_dataset(path_txt):
    """Loads dataset into memory from txt file"""
    with open(path_txt) as f:
        lines = f.readlines()
        dataset = []
        this_sentence = []
        this_tags = []
        for line in lines:
            line = line[:-1]  # delete \n
            line = line.lower()
            if line == "":
                assert len(this_sentence) == len(this_tags)
                dataset.append((this_sentence, this_tags))
                this_sentence = []
                this_tags = []
            else:
                word, pos, tag1, tag2 = line.split(" ")
                this_sentence.append(word)
                this_tags.append(tag2)
    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    train_datafile = 'data/JuJu2/train.txt'
    test_datafile = 'data/JuJu2/test.txt'
    dev_datafile = 'data/JuJu2/dev.txt'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(train_datafile)
    assert os.path.isfile(train_datafile), msg

    # Load the dataset into memory
    print("Loading juju2 dataset into memory...")
    train_data = load_dataset(train_datafile)
    test_data = load_dataset(test_datafile)
    dev_data = load_dataset(dev_datafile)
    print("- done.")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = train_data
    val_dataset = dev_data
    test_dataset = test_data

    # Save the datasets to files
    save_dataset(train_dataset, 'data/JuJu2/train')
    save_dataset(val_dataset, 'data/JuJu2/val')
    save_dataset(test_dataset, 'data/JuJu2/test')
