from re import M
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(MLP, self).__init__()

        self.lr = args.lr
        self.nepochs = args.nepochs
        self.n_inputs = input_shape
        self.n_outputs = output_shape

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, 126),
            nn.BatchNorm1d(126),
            nn.ReLU(),

            nn.Linear(126, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, self.n_outputs)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, input):
        return self.model(input)

    def fit(self, train_dataloader, val_dataloader):
        mean_train_loss = np.zeros((self.nepochs))
        mean_val_loss = np.zeros((self.nepochs))
        
        for e in range(self.nepochs):

            train_loss = []
            self.train()
            for data, labels in train_dataloader:
                self.optim.zero_grad()
                data = data.to(self.device)
                labels = labels.to(self.device)
                predictions = self.forward(data)
                loss = self.loss_function(predictions, labels)
                train_loss += [loss.item()]
                loss.backward()
                self.optim.step()
            mean_train_loss[e] = sum(train_loss) / len(train_loss)

            val_loss = []
            self.eval()
            for data, labels in val_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                predictions = self.forward(data)
                loss = self.loss_function(predictions, labels)
                val_loss += [loss.item()]
            mean_val_loss[e] = sum(val_loss) / len(val_loss)
            
            print(f"epoch {e}: \n train_loss: {mean_train_loss[e]} \t val_loss: {mean_val_loss[e]} \n" + '-'*80)

    def test(self, test_dataloader):
        test_loss = []
        correct = 0
        self.eval()
        for data, labels in test_dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            predictions = self.forward(data)
            loss = self.loss_function(predictions, labels)
            test_loss += [loss.item()]
            class_predictions = torch.argmax(predictions, axis=-1)
            correct += sum(class_predictions == labels)
        
        test_acc  = correct / len(test_dataloader.dataset)
        test_loss = sum(test_loss) / len(test_loss)

            
        print('-'*80 + '\n' +f" test_loss: {test_loss} \t test_accuracy: {test_acc * 100}% \n" + '-'*80)





