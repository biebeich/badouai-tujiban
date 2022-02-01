import re
import torch
from model import TorchModel
from loader import load_schema, load_vocab
from collections import defaultdict
from config import Config


class Predict:
    def __init__(self, config, model_path):
        self.config = config
        self.model = TorchModel(config)
        self.schema = load_schema(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.model.load_state_dict(torch.load(model_path))
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.model.eval()

    def decode(self, sentence, labels):
        labels = "".join([str(int(x)) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        return results

    def forecast(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            result = self.model(torch.LongTensor([input_id]))[0]
            result = torch.argmax(result, dim=-1)
        results = self.decode(sentence, result)
        return results


if __name__ == "__main__":
    predict = Predict(Config, "./model_output/epoch_10.pth")
    sentence = "中共中央政治局委员、书记处书记丁关根主持今天的座谈会"
    result = predict.forecast(sentence)
    result["PERSON"] += re.findall("丁关根", sentence)
    result["ORGANIZATION"] += re.findall("书记处", sentence)
    print(result)
    sentence = "延安是世界人民敬仰和向往的地方，曾接待大量外宾，周恩来指示招待外宾一定要体现艰苦奋斗的精神，要吃一点小米。"
    result = predict.forecast(sentence)
    print(result)
    result["LOCATION"] += re.findall("延安", sentence)
    print(result)