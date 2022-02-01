import json
import torch
from torch.utils.data import DataLoader

def load_schema(schema_path):
    with open(schema_path, encoding='utf8') as f:
        return json.load(f)


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict



class Datagenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.schema = load_schema(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding='utf8') as f:
            segments = f.read().split('\n\n')
            for segment in segments:
                sentence = []
                labels = []
                lines = segment.split('\n')
                for line in lines:
                    if line == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, sentence, padding=True):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token]*(self.config['max_length']-len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_data(data_path, config, shuffle=True):
    dg = Datagenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    dg = Datagenerator("./ner_data/train.txt", Config)
    print(dg[55])

