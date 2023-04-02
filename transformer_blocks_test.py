

def get_embedding(path: str):
    vocab, embedding = list(), list()
    with open(f'{path}', 'rt') as f:
        full_content = f.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embedding = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embedding.append(i_embedding)


get_embedding("./word2vec_eng/glove.6B.100d.txt")