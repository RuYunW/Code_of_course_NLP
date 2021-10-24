import numpy as np
import math
from tqdm import tqdm

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding

class Vocab:
    def __init__(self, fpath, min_count):
        self.fpath = fpath
        self.vocab_items = []
        self.token2id = {}
        self.id2token = {}
        self.min_count = min_count

        vocab_items = [VocabItem('<SOS>'), VocabItem('<UNK>'), VocabItem('<EOS>')]
        token2id = {'<SOS>': 0, '<UNK>': 1, '<EOS>': 2}
        id2token = {0: '<SOS>', 1: '<UNK>', 2: '<EOS>'}
        vocab_size = 3
        word_count = 0
        print('Vocabulary is building...')
        with open(self.fpath, 'r', encoding='utf-8') as f:
            txt_lines = f.readlines()
        for line in tqdm(txt_lines):
            vocab_items[token2id['<SOS>']].count += 1
            vocab_items[token2id['<EOS>']].count += 1
            word_count += 2
            for token in line.split(' '):
                if token in token2id:
                    id = token2id[token]
                    vocab_items[id].count += 1
                else:
                    token2id[token] = vocab_size
                    id2token[vocab_size] = token
                    vocab_items.append(VocabItem(token))
                    vocab_size += 1
                word_count += 1
        self.vocab_items = vocab_items
        self.token2id = token2id
        self.id2token = id2token
        self.word_count = word_count

        self.__sort(min_count)

    def __len__(self):
        return len(self.vocab_items)

    def __getitem__(self, item):
        return self.vocab_items[item]

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, item):
        return item in self.id2token

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<UNK>'))
        unk_id = 0
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_id].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        self.id2token = {}
        self.token2id = {}
        for id, token in enumerate(tmp):
            self.id2token[id] = token
            self.token2id[token.word] = id

        self.vocab_items = tmp

        # print
        print('Unknown vocab size: ', count_unk)
        print('Vocab size: ', len(self.vocab_items))

    def seq_indices(self, seq):
        return [self.token2id[token] if token in self.token2id else self.token2id['<UNK>'] for token in seq]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self.vocab_items)
        count = [t.count for t in self.vocab_items] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)

        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for idx, token in enumerate(self.vocab_items):
            path = []  # List of indices from the leaf to the root
            code = []  # Binary Huffman encoding from the leaf to the root

            node_idx = idx
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]
        print('Huffman Tree build successfully. ')

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab, power = 0.75):
        norm = sum([math.pow(t.count, power) for t in vocab.vocab_items])
        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0  # cumulative probability
        i = 0
        for j, unigram in tqdm(enumerate(vocab.vocab_items)):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        print('Unigram Table build successfully.')

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def save(vocab, syn0, save_path):
    dim = len(syn0[0])
    with open(save_path, 'w') as sf:
        sf.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            sf.write('%s %s\n' % (word, vector_str))
    print('Model is saved to path: \'' + save_path + '\'')

def train(fpath, min_count, dim, neg, starting_alpha, window, cbow, save_path):
    vocab = Vocab(fpath, min_count)
    syn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))
    syn1 = np.zeros(shape=(len(vocab), dim))

    if neg > 0:
        print('Unigram table is building...')
        table = UnigramTable(vocab)
    else:
        print('Huffman Tree is building...')
        vocab.encode_huffman()
    word_count = 0
    last_word_count = 0
    global_word_count = 0

    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print('Training...')
    for line in tqdm(lines):
        sent = vocab.seq_indices(['<SOS>'] + line.split(' ') + ['<EOS>'])
        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count += (word_count - last_word_count)
                last_word_count = word_count
                alpha = starting_alpha * (1 - float(global_word_count) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

            current_win = np.random.randint(low=1, high=window+1)
            context_start = max(sent_pos-current_win, 0)
            context_end = min(sent_pos+current_win+1, len(sent))
            context = sent[context_start: sent_pos] + sent[sent_pos+1: context_end]
            # CBOW
            if cbow:
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert neu1.shape[0] == dim, 'neu1 and dim do not agree'
                neu1e = np.zeros(dim)

                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)

                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * neu1
                    syn1[target] += g * neu1
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-Gram
            else:
                for context_word in context:
                    neu1e = np.zeros(dim)

                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label-p)
                        neu1e += g * syn1[target]
                        syn1[target] += g * syn0[context_word]
                    syn0[context_word] += neu1e
            word_count += 1
        global_word_count += (word_count - last_word_count)
    print('Model is saving...')
    save(vocab, syn0, save_path)
