import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model.train_data_set import get_model_input, WordDict


class DecodeDataset(Dataset):
    def __init__(self, encode_model, data_path, labels_path, set_type, word_dict, audio_length=1600):
        self.encode_model = torch.load(encode_model)
        self.data_path = data_path + "/" + set_type + "/wav/"
        self.labels_path = labels_path + "/" + set_type + "_labels.txt"
        self.w2i, _ = word_dict.get_dict()
        self.wav_paths = []
        self.phonetic_list = []
        self.wav_encodes = []
        self.label_encodes = []

        with open(self.labels_path, 'r') as f:
            for line in f:
                item = line.rstrip('\n').split('\t')
                self.wav_paths.append(self.data_path + item[0])
                self.phonetic_list.append(item[1].replace('\n', ''))

        print("开始音频编码")
        for i in tqdm(range(len(self.wav_paths))):
            wav_feature = get_model_input(self.wav_paths[i], audio_length)
            decode_input = self.encode_model.forward(wav_feature.unsqueeze(0))
            decode_input = decode_input.reshape((decode_input.shape[0], -1)).detach().numpy()
            self.wav_encodes.append(decode_input)

            phonetics = self.phonetic_list[i].split(' ')
            label_encode = [self.w2i[phonetic] for phonetic in phonetics]
            self.label_encodes.append(label_encode)

        self.data_len = len(self.wav_encodes)
        # self.label_num = len(self.w2i)
        self.label_num = 1922

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return_labels_data = np.zeros((64,), dtype=np.int16)

        wav_encode = self.wav_encodes[item]
        labels = self.label_encodes[item]

        return_labels_data[0:len(labels)] = labels
        return wav_encode, return_labels_data


class DecodeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DecodeModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wordDict = WordDict(["../data/train_labels.txt", "../data/test_labels.txt"])
    dataset = DecodeDataset("../model/model.pt", "../../data", "../data", "train", wordDict)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DecodeModel(input_size=1922, hidden_size=128, output_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for batch_idx, sample in enumerate(loader):
            input_data, labels_data = sample
            input_data = input_data.type(torch.FloatTensor).to(device)
            labels_data = labels_data.type(torch.FloatTensor).to(device)

            model.zero_grad()
            output = model(input_data)

            loss = criterion(output, labels_data)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Train Loss: {total_loss / len(dataset):.3f}")

    torch.save(model, "../../model/decode.pt")
