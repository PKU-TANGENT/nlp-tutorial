import os
from torch.utils.data import Dataset

def build_vocab(config):
    word2id = {'PAD': 0}
    word_cnt = 1
    id2word = ['PAD']
    with open(os.path.join(config.data_path, config.train_file), 'r', encoding='utf-8') as fr:
        raw = fr.readlines()
    for item in raw:
        if item[0] == '\n':
            continue
        t = item[0]
        if t not in word2id:
            word2id[t] = word_cnt
            word_cnt += 1
            id2word.append(t)
    word2id['UNK'] = word_cnt
    word_cnt += 1
    id2word.append('UNK')
    assert len(word2id) == len(id2word)
    return word2id, id2word, word_cnt


class PeopleDailyDataset(Dataset):
    def __init__(self, word2id, tag2id, data_path):
        super(PeopleDailyDataset, self).__init__()
        self.data_path = data_path
        self.word2id = word2id
        self.tag2id = tag2id
        self.raw = None
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            self.raw = fr.readlines()

        self.x = []
        self.y = []
        self.lengths = []
        self.preprocess()


    def preprocess(self):
        xx, yy = [], []
        for item in self.raw:
            if item[0] == '\n':
                if len(xx) == 0:
                    continue
                assert len(xx) == len(yy)
                self.x.append(xx)
                self.y.append(yy)
                self.lengths.append(len(xx))
                xx, yy = [], []
                continue
            item = item.strip('\n')
            if item[0] in self.word2id:
                xx.append(self.word2id[item[0]])
            else:
                xx.append(self.word2id['UNK'])
            yy.append(self.tag2id[item[2:]])


    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index]