import numpy as np
from torch.utils.data import Dataset
from itertools import repeat


def get_embedding(path: str, pad=False, unk=False):
    vocab, embeddings = dict(), list()
    with open(f'{path}', 'rt') as f:
        full_content = f.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embedding = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab[i_word] = i
        embeddings.append(i_embedding)
    embs_npa = np.array(embeddings)
    if pad:
        vocab['<pad>'] = len(vocab)
        pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
        embs_npa = np.vstack((embs_npa, pad_emb_npa))
    if unk:
        vocab['<unk>'] = len(vocab)
        unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)
        embs_npa = np.vstack((embs_npa, unk_emb_npa))
    return vocab, embs_npa


class TextDataset(Dataset):
    def __init__(self, vocab2index, data):
        self.data = data
        self.vocab2index = vocab2index
        self.max_len = self._get_max_len()
        
    def _get_max_len(self):
        sentence_list = [x.split() for x in self.data]
        len_list = (len(x) for x in sentence_list)
        return max(len_list)
        
    def _text_to_index(self, text):
        indexes = np.zeros((self.max_len,))
        text = text.lower().split()
        indexes[:len(text)] = np.array([self.vocab2index[x] if x in self.vocab2index else self.vocab2index["<unk>"] for x in text])
        pad = list()
        pad.extend(repeat(self.vocab2index["<pad>"],self.max_len - len(text)))
        indexes[len(text):] = pad
        indexes = indexes.dtype(np.int32)
        return indexes
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        text = self.data[ind]
        indexes = self._text_to_index(text)
        return indexes


def load_data(train_path):
    with open(f'{train_path}/train.en') as f:
        train_data_x = f.read().strip().split('\n')
    return train_data_x

train_data_x = load_data("./train")
eng_vocab, eng_emb = get_embedding("./word2vec_eng/glove.6B.100d.txt", pad=True)
dataset = TextDataset(eng_vocab, train_data_x)
dataset[0]