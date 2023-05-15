from model.speech_model import SpeechModel


def train_save(data_path, labels_path, model_path, train_epoch=100, train_batch=4):
    speech_model = SpeechModel(data_path, labels_path, model_path, train_epoch=train_epoch, train_batch=train_batch)
    speech_model.train()
    speech_model.export()
