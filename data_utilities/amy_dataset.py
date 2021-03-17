from torch.utils.data import Dataset


class AmyDataset(Dataset):
    def __init__(self, x, L):
        self.x = x
        self.L = L

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.L[index]


class Datasetv2(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Datasetv3(Dataset):
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.t[index]


class Datasetv4(Dataset):
    def __init__(self, x, y, z, a):
        self.x = x
        self.y = y
        self.z = z
        self.a = a

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index], self.a[index]


class Datasetv5(Dataset):
    def __init__(self, x, y, z, a, t):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.t = t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index], self.a[index], self.t[index]