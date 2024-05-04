import pickle as pkl
from dataset.dataset import SaladsDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter

num_epochs = 1000


def train(batch):
    pass


if __name__ == "__main__":
    with open("../data/50salads_one_hot.pkl", "rb") as f:
        dataset = pkl.load(f)

    train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, shuffle=True, random_state=17)
    train_salads = SaladsDataset(train_dataset)
    test_salads = SaladsDataset(test_dataset)

    train_loader = DataLoader(train_salads, batch_size=10,
                              collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1))
    test_loader = DataLoader(test_salads, batch_size=10,
                             collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=-1))

    for epoch in range(num_epochs):
        for B in train_loader:
            x = B[:, :, :2]
            y = B[:, :, 2:]
