import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.device = config.device

        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.tagset_size = len(self.tag2id)

        # map input tokens to unique embedding(vector)
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        # LSTM is a variant of Recurrent Neural Network(RNN)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # linear layer, predict the probability of each tag
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        # Loss: compute the distance between our prediction and the gold tag
        self.loss = CrossEntropyLoss()

    def forward(self, sent, labels, lengths, mask):
        embedded = self.emb(sent)

        # The padded batch should be packed before LSTM
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        # The packed batch should be padded after LSTM
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # lstm_out: [batch_size, max_len, hidden_dim]
        logits = self.hidden2tag(lstm_out)  # logits: [batch_size, max_len, tagset_size]

        # Predict the tags
        pred_tag = torch.argmax(logits, dim=-1)

        # Compute loss. Pad token must be masked before computing the loss.
        logits = logits.view(-1, self.tagset_size)[mask.view(-1) == 1.0]
        loss = self.loss(logits, labels.view(-1))

        return loss, pred_tag
