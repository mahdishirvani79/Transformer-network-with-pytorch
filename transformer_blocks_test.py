import numpy as np

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

vi_vocab, vi_emb = get_embedding("./word2vec_vi/word2vec_vi_words_100dims.txt")