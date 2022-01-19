import math
import jieba
from collections import defaultdict


class NgramLanguageModel:
    def __init__(self, corpus=None, n=3):
        self.n = n
        self.corpus = corpus
        self.sep = "🗡"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.unk_prob = 1e-5
        self.fix_backoff_prob = 0.4
        self.ngram_count_dict = dict((x, defaultdict(int)) for x in range(1, self.n+1))
        self.ngram_count_prob_dict = dict((x, defaultdict(int)) for x in range(1, self.n+1))
        self.ngram_count(corpus)
        self.calc_ngram_prob()

    def sentence_segment(self, sentence):
        return list(sentence)

    def ngram_count(self, corpus):
        for sentence in corpus:
            wordlists = self.sentence_segment(sentence)
            wordlists = [self.sos] + wordlists + [self.eos]
            for window_size in range(1, self.n+1):
                for index, word in enumerate(wordlists):
                    if len(wordlists[index:index + window_size]) != window_size:
                        continue
                    ngram = self.sep.join(wordlists[index: index+window_size])
                    self.ngram_count_dict[window_size][ngram] += 1
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        return self.ngram_count_dict

    def calc_ngram_prob(self):
        for window_size in range(1, self.n+1):
            for ngram, count in self.ngram_count_dict[window_size].items():
                if window_size > 1:
                    prefix_ngram = self.sep.join(ngram.split(self.sep)[:-1])
                    prefix_ngram_count = self.ngram_count_dict[window_size-1][prefix_ngram]
                else:
                    prefix_ngram_count = self.ngram_count_dict[0]
                self.ngram_count_prob_dict[window_size][ngram] = count / prefix_ngram_count
        return

    def get_ngram_prob(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1:
            return self.unk_prob
        else:
            ngram = self.sep.join(ngram.split(self.sep)[1:])
            return self.fix_backoff_prob*self.get_ngram_prob(ngram)

    def calc_ppl(self, sentence):
        wordlist = self.sentence_segment(sentence)
        wordlist = [self.sos] + wordlist + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(wordlist):
            ngram = self.sep.join(wordlist[max(0, index-self.n+1): index+1])
            ngram_prob = self.get_ngram_prob(ngram)
            sentence_prob += math.log(ngram_prob)
        return 2**(-sentence_prob/len(wordlist))

    def predict(self, sentence):
        wordlist = self.sentence_segment(sentence)
        wordlist = [self.sos] + wordlist + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(wordlist):
            ngram = self.sep.join(wordlist[max(0, index - self.n + 1): index + 1])
            ngram_prob = self.get_ngram_prob(ngram)
            sentence_prob += math.log(ngram_prob)
        return sentence_prob


if __name__ == "__main__":
    corpus = open("财经.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    print("词总数:", lm.ngram_count_dict[0])
    # print(lm.ngram_count_prob_dict)
    print(lm.predict("萦绕在世界经济的阴霾仍未"))





