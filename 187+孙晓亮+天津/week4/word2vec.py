import jieba
from gensim.models import Word2Vec


def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def main():
    sentences = []
    with open("corpus.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line))
    train_word2vec_model(sentences, 100)
    return


if __name__ == "__main__":
    main()
