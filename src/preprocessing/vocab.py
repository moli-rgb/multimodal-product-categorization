from collections import Counter
from src.preprocessing.text_cleaner import clean_text


class Vocabulary:
    def __init__(self, texts, min_freq=2):
        self.texts = texts
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab()

    def build_vocab(self):
        all_words = [word for title in self.texts for word in title.split()]
        word_counts = Counter(all_words)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text):
        words = text.split()
        ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        return ids

    def __len__(self):
        return len(self.word2idx)

