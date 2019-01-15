"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pickle
import bcolz

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--use_glove', default=0, help="use pretrain embedding glove", type=int)
parser.add_argument('--glove_emb_dim', default=300, help="pretrain embedding glove embedding dim", type=int)


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.data[0])
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.learning_rate * pow(0.9, epoch)

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
        logging.info("- validation dataset")
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)

        # Evaluate for one epoch on test set
        num_steps = (params.test_size + 1) // params.batch_size
        test_data_iterator = data_loader.data_iterator(test_data, params, shuffle=False)
        logging.info("- test dataset")
        test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path        
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val', 'test'], args.data_dir)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    if args.use_glove:
        logging.info("use glove pretrained embedding, embedding dim: {}".format(args.glove_emb_dim))
        words = []
        with open(os.path.join(args.data_dir, "words.txt"), "r") as f_words:
            lines = f_words.readlines()
            for line in lines:
                words.append(line[:-1])
        glove_vectors = bcolz.open(f'glove.6B.50d.dat')[:]
        glove_words = pickle.load(open(f'glove.6B.50_words.pkl', 'rb'))
        glove_word2idx = pickle.load(open(f'glove.6B.50_idx.pkl', 'rb'))

        glove = {w: glove_vectors[glove_word2idx[w]] for w in glove_words}
        matrix_len = len(words)
        weights_matrix = np.zeros((matrix_len, args.glove_emb_dim))
        words_found_count = 0
        words_not_found = []
        for i, word in enumerate(words):
            try:
                weights_matrix[i] = glove[word]
                words_found_count += 1
            except KeyError:
                words_not_found.append(word)
                weights_matrix[i] = np.random.normal(scale=0.6, size=(args.glove_emb_dim,))
        logging.info("words found in glove: {}, total verb size: {}, converage: {}".
                     format(words_found_count, len(words), words_found_count / len(words)))
        with open("not_found.txt", "w") as f:
            for each in words_not_found:
                f.write(each + "\n")

    # Define the model and optimizer
    pre_embedding_weight = weights_matrix if args.use_glove else None
    model = net.Net(params, pre_embedding_weight).cuda() if params.cuda else net.Net(params, pre_embedding_weight)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, test_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
