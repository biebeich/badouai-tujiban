import json
import copy
from ngram_language_model import NgramLanguageModel


class Corrector:
    def __init__(self, language_model):
        self.language_model = language_model
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        self.threshold = 8

    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding='utf8') as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    def get_candidate_sentence_prob(self, candidates, char_list, index):
        if candidates == []:
            return [-1]
        result = []
        for char in candidates:
            char_list[index] = char
            sentence = "".join(char_list)
            sentence_prob = self.language_model.predict(sentence)
            sentence_prob -= self.sentence_prob_baseline
            result.append(sentence_prob)
        return result

    def correction(self, string):
        char_list = list(string)
        fix = {}
        self.sentence_prob_baseline = self.language_model.predict(string)
        for index, char in enumerate(char_list):
            candidates = self.sub_dict.get(char, [])
            candidate_probs = self.get_candidate_sentence_prob(candidates, copy.deepcopy(char_list), index)
            if max(candidate_probs) > self.threshold:
                sub_char = candidates[candidate_probs.index(max(candidate_probs))]
                print("第%d个字建议修改： %s -> %s, 概率提升：%f" % (index, char, sub_char, max(candidate_probs)))
                fix[index] = sub_char
        char_list = [fix[i] if i in fix else char for i, char in enumerate(char_list)]
        return "".join(char_list)


corpus = open("财经.txt", encoding='utf8').readlines()
lm = NgramLanguageModel(corpus, 3)
cr = Corrector(lm)
string = "美联储可能才取新的措施来提针每国经济"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)
