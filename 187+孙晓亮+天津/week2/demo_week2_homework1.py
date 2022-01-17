import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from torch.nn import functional


class TorchModel(nn.Module):
    def __init__(self, vocab, char_dim, sentence_length):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, char_dim)
        self.layer = nn.Linear(char_dim, char_dim)
        self.dropout = nn.Dropout(0.1)
        self.activate = torch.sigmoid
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(char_dim, 3)
        self.loss = functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.layer(x)
        x = self.dropout(x)
        x = self.activate(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        x = self.classify(x)
        y_pred = self.activate(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab["UNK"] = len(vocab)+1
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc") & set(x) and not set("rst") & set(x):
        y = 1
    elif not set("abc") & set(x) and set("rst") & set(x):
        y = 2
    else:
        y = 0
    x = [vocab.get(word, vocab["UNK"]) for word in x]
    return x, y


def build_dataset(vocab, sample_length, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def load_model(vocab, char_dim, sentence_length):
    load_model = TorchModel(vocab, char_dim, sentence_length)
    return load_model


def evaluate(model, vocab, evaluate_sample, sentence_length):
    model.eval()
    x, y = build_dataset(vocab, evaluate_sample, sentence_length)
    print("本次预测集中共有%d个正样本， %d个负样本。" % (sum(y), evaluate_sample-sum(y)))
    correct, wrongs = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrongs += 1
    print("正确预测个数：%d， 正确率：%f" % (correct, correct/evaluate_sample))
    return correct/(correct + wrongs)


def main():
    epoch_num = 10
    train_sample = 1000
    batch_size = 20
    evaluate_sample = 200
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    vocab = build_vocab()
    model = load_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(vocab, batch_size, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=================\n第%d轮平均loss：%f:" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, evaluate_sample, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log])
    plt.plot(range(len(log)), [l[1] for l in log])
    plt.show()

    torch.save(model.state_dict(), "model.pth")

    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = load_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char) for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(torch.argmax(result[i]), input_string, result[i])

if __name__ == "__main__":
    main()
    test_strings = ["abvxee", "tdsdfg", "rqweqg", "nlkdww"]
    predict("model.pth", "vocab.json", test_strings)

