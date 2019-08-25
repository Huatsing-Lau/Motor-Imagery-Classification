import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import tation_get

train_size = 2880
test_size = 720
all_buff_data, all_label_data, sample_cont = tation_get.get_dataset()
all_train_data = np.zeros([2880, 64, 640])
train_label_data = np.zeros(2880)
all_test_data = np.zeros([720, 64, 640])
test_label_data = np.zeros(720)

all_train_data = all_buff_data[0:2880,:,:]
all_test_data = all_buff_data[2880:3600,:,:]
train_label_data = all_label_data[0:2880]
test_label_data = all_label_data[2880:3600]
for sub in range(240):
    all_train_data[sub * 12: (sub + 1) * 12, :, :] = all_buff_data[sub * 15: sub * 15 + 12, :, :]
    train_label_data[sub * 12: (sub + 1) * 12] = all_label_data[sub * 15: sub * 15 + 12]
    all_test_data[sub * 3 : (sub + 1) * 3, :, :] = all_buff_data[sub * 15 + 12: sub * 15 + 15, :, :]
    test_label_data[sub * 3 : (sub + 1) * 3] = all_label_data[sub * 15 + 12: sub * 15 + 15]
all_train_data = all_train_data.transpose(0, 2, 1)
all_test_data = all_test_data.transpose(0,2,1)
print(train_label_data.shape)
print(test_label_data.shape)
train_zero_label = 0;
train_one_label = 0;
test_zero_label = 0;
test_one_label = 0;
train_buff_tensor = torch.from_numpy(all_train_data)
test_buff_tensor = torch.from_numpy(all_test_data)
train_label_tensor = torch.from_numpy(train_label_data)
test_label_tensor = torch.from_numpy(test_label_data)
train_dataset = Data.TensorDataset(train_buff_tensor, train_label_tensor)
test_dataset = Data.TensorDataset(test_buff_tensor, test_label_tensor)

test_dataloader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False
)

train_dataloader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)


class LSTM_EEG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lstm_1 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm_1(x)

        out = self.output(x[:, -1, :])
        return out


lstm_model = LSTM_EEG()
#lstm_model.load_state_dict(torch.load("model_test_new_237.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm_model.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.Adam(lstm_model.parameters())
counter = 0
mat_one = 0
for step, (x_test_batch, y_test_batch) in enumerate(test_dataloader):
    test_output = lstm_model(x_test_batch.float())
    y_test_batch = y_test_batch.float()
    _, prediction = torch.max(test_output.data, 1)
    counter += (y_test_batch == prediction.float()).sum().item()
print("accuracy: " + str(counter / test_size * 100) + "%")
for epoch in range(300):
    running_loss = 0.0
    for step, (x_batch, y_batch) in enumerate(train_dataloader):
        print("epoch: " + str(epoch) + " batch: " + str(step))
        optimizer.zero_grad()
        output_result = lstm_model(x_batch.float())
        loss = criterion(output_result, y_batch.long())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("epoch:" + str(epoch) + " loss:" + str(running_loss / train_size * 100))
    torch.save(lstm_model.state_dict(), "model_test_new_" + str(epoch) + ".pth")
    model_new = LSTM_EEG()
    model_new.load_state_dict(torch.load("./model_test_new_" + str(epoch) + ".pth"))
    counter = 0
    test_loss = 0.0
    for step, (x_test_batch, y_test_batch) in enumerate(test_dataloader):
        print("epoch: " + str(epoch) + " test_batch: " + str(step))
        test_output = model_new(x_test_batch.float())
        test_loss += criterion(test_output, y_test_batch.long())
        y_test_batch = y_test_batch.float()
        _, prediction = torch.max(test_output.data, 1)
        counter += (y_test_batch == prediction.float()).sum().item()
    print("test_loss: " + str(test_loss.item() / test_size * 100) + " accuracy: " + str(counter / test_size))