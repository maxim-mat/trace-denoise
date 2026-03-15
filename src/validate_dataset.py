import pickle as pkl
from dataset.dataset import SaladsDataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--device', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.path, 'rb') as f:
        data = pkl.load(f)

    dataset = SaladsDataset(data['target'], data['stochastic'])
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)

    for i, (x, y) in tqdm(enumerate(loader)):
        # Assuming 'y' is your target tensor
        x = x.to(args.device)
        x = torch.argmax(x, dim=2)
        # Check if labels are within the expected range for your model
        # Replace 'num_classes' with your actual model output dimension
        if (x >= args.n_classes).any() or (x < 0).any():
            print(f"ERROR: Batch {i} contains invalid labels!")
            print(f"Max label: {x.max()}, Min label: {x.min()}")

    print("successfully validated dataset")
