import json
import numpy as np
import torch
from config import Config
from model import SiameseNetwork


class QASystem:
    def __init__(self, know_base_path, model_path, vocab_path):
        self.load_model(model_path, vocab_path)
        self.load_know_base(know_base_path)
        print("知识库加载完毕，可以开始问答！")

    def load_know_base(self, know_base_path):
        self.know_base_question_vectors = []
        self.index_to_target = {}
        question_index = 0
        with open(know_base_path, encoding='utf8') as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                for question in questions:
                    self.index_to_target[question_index] = target
                    question_index += 1
                    vector = self.string_to_vector(question)
                    self.know_base_question_vectors.append(vector)
        self.know_base_question_vectors = np.array(self.know_base_question_vectors)
        return

    def load_model(self, model_path, vocab_path):
        self.vocab = self.load_vocab(vocab_path)
        Config["vocab_size"] = len(self.vocab)
        self.model = SiameseNetwork(Config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        return

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding='utf8') as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        return token_dict

    def string_to_vector(self, string):
        input_id = []
        for char in string:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        vector = self.model(torch.LongTensor([input_id]))
        vector = vector.cpu().detach().numpy()
        vector /= np.sqrt(np.sum(np.square(vector)))
        return vector

    def find_most_similar_vector(self, question_vector):
        similaritys = np.dot(question_vector, self.know_base_question_vectors.T)
        return np.argmax(similaritys)

    def query(self, question):
        question_vector = self.string_to_vector(question)
        most_similar_vector_index = self.find_most_similar_vector(question_vector)
        target = self.index_to_target[most_similar_vector_index]
        return target


if __name__ == "__main__":
    qas = QASystem("train.json", "model_output/epoch_10.pth", "chars.txt")
    while True:
        question = input("请输入问题：")
        res = qas.query(question)
        print("命中问题：", res)
        print("---------------------------")