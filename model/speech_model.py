import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model.conformer import Conformer
from util.dict import WordDict
from model.train_data_set import SpeechData, collate_fn


class SpeechModel(object):
    def __init__(self,
                 data_path,
                 labels_path,
                 model_path,
                 input_dim=80,
                 encoder_dim=32,
                 num_encoder_layers=3,
                 train_epoch=10,
                 train_batch=4,
                 train_lr=0.0001):
        super(SpeechModel, self).__init__()
        self.train_epoch = train_epoch
        self.train_lr = train_lr
        self.loss_list = []
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phone_dict = WordDict([labels_path + "/train_labels.txt", labels_path + "/test_labels.txt"])
        self.i2w = phone_dict.get_i2w()
        self.model = Conformer(num_classes=len(self.i2w)+1,
                               input_dim=input_dim,
                               encoder_dim=encoder_dim,
                               num_encoder_layers=num_encoder_layers).to(self.device)
        train_dataset = SpeechData(data_path, labels_path, "train", phone_dict)
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=collate_fn, shuffle=True)
        print("训练初始化完成！")

    def get_i2w(self):
        return self.i2w

    def train(self):
        print(f"使用\t{self.device}\t进行训练")
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_lr)
        criterion = nn.CTCLoss().to(self.device)
        for epoch in range(self.train_epoch):
            epoch_loss = 0.0
            for batch_idx, samples in enumerate(self.train_loader):
                optimizer.zero_grad()
                input_data, labels_data, input_lengths, label_lengths = samples
                input_data = input_data.to(self.device)
                input_lengths = input_lengths.to(self.device)
                labels_data = labels_data.to(self.device)
                label_lengths = label_lengths.to(self.device)
                outputs, output_lengths = self.model(input_data, input_lengths)
                loss = criterion(outputs.transpose(0, 1), labels_data, output_lengths, label_lengths)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"epoch: {epoch+1}\t loss: {epoch_loss/len(self.train_loader):.5f}")
            self.loss_list.append(epoch_loss/len(self.train_loader))

    def export(self):
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        self.model.to(torch.device('cpu'))
        jit_model = torch.jit.script(self.model)
        jit_model.save(self.model_path + "/jit_model.pt")
        torch.save(self.model, self.model_path + "/model.pt")
        print("模型保存成功！")
