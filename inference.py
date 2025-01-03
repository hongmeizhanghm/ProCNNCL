from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import pdb
import csv


class ProCNNCL(nn.Module):
    def __init__(self,
                 target_shape=5120,
                 latent_dimension=5120,
                 latent_activation=nn.ReLU, **config):
        super(ProCNNCL, self).__init__()
        self.target_shape = target_shape
        mlp_hidden_dim = config["hidden_dim"]
        mlp_out_dim = config["out_dim"]
        out_binary = config["binary"]
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)
        self.target_projector2 = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector2[0].weight)

        self.target_projector3 = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector3[0].weight)

        self.mlp_classifier = MLPDecoder(latent_dimension, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, features):
        features = features.to(torch.float32).cuda()
        target_projection = self.target_projector(features)
        target_projection2 = self.target_projector2(target_projection)
        target_projection3 = self.target_projector2(target_projection2)
        return F.normalize(target_projection3, dim=-1)

    def train_model(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return self.mlp_classifier(output)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def list_files_recursively(path):
    all_files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files_path.append(file_path)
    return all_files_path, files


class NewDataset(Dataset):
    def __init__(self, files_path, file_names):
        super().__init__()
        self.files = file_names
        self.file_path = files_path

    def __len__(self):
        return len(self.files)

    def get_embedding(self, files):
        Xs = []
        embs = torch.load(files)
        token_representations = embs['representations'][33]
        sequence_representations = torch.mean(token_representations, 0)
        Xs.append(sequence_representations)
        Xs = torch.stack(Xs, dim=0).numpy()
        return Xs

    def __getitem__(self, index):
        sub_files = self.file_path[index]
        sub_names = self.files[index]
        embedding = self.get_embedding(sub_files)
        return sub_names, embedding

def createDict(*args):
    print(args)
    return dict(((k, eval(k)) for k in args))


num_filters = [512, 254, 128]
embed_dim = 128
kernel_size = [3, 5, 7]
padding = True
in_dim = 129536
hidden_dim = 512
out_dim = 128
binary = 1

files_path, file_names = list_files_recursively("./data_embedding") #Embeddings obtained through ESM-2

newdata = NewDataset(files_path, file_names)

prediction_dataloader = DataLoader(newdata, batch_size=len(newdata), shuffle=False, drop_last=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './salt_tz/salt-alkali.pth'  # The path of the model
cfg = createDict("num_filters", "embed_dim", "kernel_size", "padding", "in_dim",
                 "hidden_dim", "out_dim", "binary")
model = ProCNNCL(**cfg).to(device)
model.load_state_dict(torch.load(model_path))

threshold = 0.9926 # The thresholds corresponding to the model


all_res = []
with torch.no_grad():
    model.eval()
    for file_names, embeddings in prediction_dataloader:
        embedding = torch.tensor(embeddings).squeeze(1).to(device)
        out = model.train_model(embedding)
        predictions = torch.sigmoid(out).squeeze().cpu()
        indexes = (predictions >= threshold).nonzero().cpu()
        gene_names = np.array(file_names)
        salt_genes = gene_names[indexes]
        p_values = predictions[indexes]
        shorten_names = [element[:-3] for sublist in salt_genes.tolist() for element in sublist]
        res = [res for res in zip(shorten_names, [value for p_values in p_values.tolist() for value in p_values])]
        all_res.extend(res)
    print("In total, there are ", len(newdata), "genes")
    print(len(all_res), " are found to be resistance genes")

    with open("./resistance_genes.csv", 'w') as f:
         writer = csv.writer(f)
         writer.writerows(all_res)
