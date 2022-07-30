import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from preprocess_data import build_vocab, PeopleDailyDataset
from BiLSTM_crf import BiLSTM
from utils import *


def train(args, model, train_features, val_features):
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.zero_grad()

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, shuffle=True, drop_last=True, )

    num_step = 0
    for epoch in range(args.num_epochs):
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()   # Switch to the train mode
            input_ids, labels, labels_flatten, lengths, input_mask = batch

            # Move inputs to the same device of our model.
            input_ids = torch.tensor(input_ids).to(args.device)
            labels_flatten = torch.tensor(labels_flatten).to(args.device)
            input_mask = torch.tensor(input_mask)

            loss, _ = model(input_ids, labels_flatten, lengths, input_mask)

            # Update the parameters of the model
            loss.backward()
            optimizer.step()

            # The gradient of the model should be zero before next batch
            model.zero_grad()

            num_step += 1
            if num_step % args.eval_steps == 0 and num_step != 0:
                res = evaluate(args, model, val_features)
                print('Epoch: {:3} | Step: {:6} | Loss: {:8.4f} | P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(
                    epoch + 1, num_step, loss, res['p'], res['r'], res['f1']))
                torch.save(model.state_dict(), args.model_path) # save the model


def evaluate(args, model, val_features):
    val_dataloader = DataLoader(val_features, batch_size=args.test_batch_size, collate_fn=collate_fn)
    metric = Metric(args.id2word, args.id2tag)
    model.eval()    # Switch to the evaluate mode
    for batch in val_dataloader:
        input_ids, labels, labels_flatten, lengths, input_mask = batch

        input_ids = torch.tensor(input_ids).to(args.device)
        labels_flatten = torch.tensor(labels_flatten).to(args.device)
        input_mask = torch.tensor(input_mask)

        _, pred = model(input_ids, labels_flatten, lengths, input_mask)

        for i in range(len(labels)):
            metric.add(input_ids[i, :lengths[i]].cpu().numpy().tolist(),
                       pred[i, :lengths[i]].cpu().numpy().tolist(),
                       labels[i])

    p, r, f1 = metric.get()
    return {
        'p': p * 100,
        'r': r * 100,
        'f1': f1 * 100
    }


def test(args, model, test_features):
    test_dataloader = DataLoader(test_features, batch_size=args.test_batch_size, collate_fn=collate_fn)
    metric = Metric(args.id2word, args.id2tag)
    model.eval()
    for batch in test_dataloader:
        input_ids, labels, labels_flatten, lengths, input_mask = batch

        input_ids = torch.tensor(input_ids).to(args.device)
        labels_flatten = torch.tensor(labels_flatten).to(args.device)
        input_mask = torch.tensor(input_mask)

        _, pred = model(input_ids, labels_flatten, lengths, input_mask)

        for i in range(len(labels)):
            metric.add(input_ids[i, :lengths[i]].cpu().numpy().tolist(),
                       pred[i, :lengths[i]].cpu().numpy().tolist(),
                       labels[i])

        input_ids = input_ids.cpu().numpy().tolist()
        for i in range(len(labels)):
            sent = []
            pred_label = []

            for j in range(lengths[i]):
                sent.append(args.id2word[input_ids[i][j]])
                pred_label.append(args.id2tag[pred[i][j]])

    p, r, f1 = metric.get()
    print('Test over. P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(p * 100, r * 100, f1 * 100))



def main():
    # Parse the cmd arguments
    parser = argparse.ArgumentParser()

    # Path and file configs
    parser.add_argument('--data_path', default='./data/renMinRiBao', help='The dataset path.', type=str)
    parser.add_argument('--model_path', default='./model/bilstm.pt', help='The model will be saved to this path.', type=str)
    parser.add_argument('--train_file', default='train_data.txt', type=str)
    parser.add_argument('--val_file', default='val_data.txt', type=str)
    parser.add_argument('--test_file', default='test_data.txt', type=str)
    parser.add_argument('--tag2id_file', default='tags.txt', type=str)

    # Model configs
    parser.add_argument('--emb_dim', default=128, help='Tokens will be embedded to a vector.', type=int)
    parser.add_argument('--hidden_dim', default=256, help='The hidden state dim of BiLSTM.', type=int)

    # Optimizer config
    parser.add_argument('--lr', default=1e-4, help='Learning rate of the optimizer.', type=float)

    # Training configs
    parser.add_argument('--train_batch_size', default=16, help='Batch size for training.', type=int)
    parser.add_argument('--test_batch_size', default=16, help='Batch size for testing.', type=int)
    parser.add_argument('--num_epochs', default=10, help='Batch size for training.', type=int)
    parser.add_argument('--eval_steps', default=200, help='Total number of training epochs to perform.', type=int)

    # Device config
    parser.add_argument('--gpu', default=0, type=int)

    # Mode config
    parser.add_argument('--test', help='Test on the testset.', action='store_true')

    args = parser.parse_args()

    # Specify the device. If you has a GPU, the training process will be accelerated.
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Every word must be mapped to a unique index
    word2id, id2word, word_cnt = build_vocab(args)
    args.vocab_size = word_cnt
    args.id2word = id2word

    tag2id = {}
    id2tag = []
    with open(os.path.join(args.data_path, args.tag2id_file), 'r') as fr:
        for i, item in enumerate(fr.readlines()):
            tag2id[item.strip('\n')] = i
            id2tag.append(item.strip('\n'))
    args.tag2id = tag2id
    args.id2tag = id2tag

    # Load the datasets
    train_features = PeopleDailyDataset(word2id, tag2id, os.path.join(args.data_path, args.train_file))
    val_features = PeopleDailyDataset(word2id, tag2id, os.path.join(args.data_path, args.val_file))
    test_features = PeopleDailyDataset(word2id, tag2id, os.path.join(args.data_path, args.test_file))


    model = BiLSTM(args)
    model = model.to(args.device)   # move the model into GPU if GPU is available.

    if not args.test:
        train(args, model, train_features, val_features)
    else:
        model.load_state_dict(torch.load(args.model_path))
        test(args, model, test_features)


if __name__ == '__main__':
    main()