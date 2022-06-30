import torch
import torch.utils
import torch.utils.data
import torch.optim
import torch.autograd
from feature import extractFeatureWithLabel
from tqdm import tqdm

class MyCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(2,32,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.flatten = torch.nn.Flatten()
        self.full1 = torch.nn.Linear(13*13*64, 1000)
        self.full2 = torch.nn.Linear(1000, 2)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.full1(x)
        x = self.full2(x)
        return x

def main(wavelet_func: str):
    MY_EPOCH = 2

    filename = "data/record_07_28.csv"
    features, labels = extractFeatureWithLabel(filename, wavelet_func)
    t_features = torch.from_numpy(features).double()
    t_labels = torch.from_numpy(labels).long()
    assert t_features.shape[0] == t_labels.shape[0]
    size = t_features.shape[0]

    my_dataset = torch.utils.data.TensorDataset(t_features, t_labels)
    train_set, test_set = torch.utils.data.random_split(my_dataset, [int(0.7*size), size-int(0.7*size)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)

    my_cnn = MyCNN()
    my_opt = torch.optim.Adam(my_cnn.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(MY_EPOCH):
        for i, (x, y) in tqdm(enumerate(train_loader), desc="Epoch {}".format(epoch)):
            batch_x = torch.autograd.Variable(x)
            batch_y = torch.autograd.Variable(y)
            out_y = my_cnn(batch_x)
            loss = loss_func(out_y, batch_y)
            my_opt.zero_grad()
            loss.backward()
            my_opt.step()
    
    torch.save(my_cnn, "model/mycnn_{}.pt".format(wavelet_func))

    trained_model = torch.load("model/mycnn_{}.pt".format(wavelet_func))
    acc_sum = []
    for i, (test_x, test_y) in  enumerate(test_loader):
        x = torch.autograd.Variable(test_x)
        y = torch.autograd.Variable(test_y)
        out = trained_model(x)

        acc = torch.max(out, 1)[1].numpy() == y.numpy()
        acc_sum.append(acc.mean())
    print("accurancy for {}: {}".format(wavelet_func, sum(acc_sum)/len(acc_sum)))

if __name__ == "__main__":
    # wavelets = ["cgau1","cgau2","cgau3","cgau4","gaus1","gaus2","gaus3","gaus4","mexh","morl"]
    wavelets = ["gaus1"]
    for wavelet_func in wavelets:
        main(wavelet_func)