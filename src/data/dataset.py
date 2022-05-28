from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):

    def __init__(self, meta_df, seq_length):
        self.meta_df = meta_df
        self.seq_length = seq_length

    def __len__(self):
        return len(self.meta_df) - self.seq_length - 1

    def __getitem__(self, index):
        X = self.meta_df.iloc[index: index + self.seq_length, 1:].to_numpy().astype("float32")
        y = self.meta_df.iloc[index + self.seq_length, 1:].to_numpy().astype("float32")

        return X, y


def split_data(dataframe, split_size=0.8):
    df = dataframe.copy()
    split_index = int(split_size * len(df))
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]

    return train_df, val_df


def create_loaders(train_set, val_set, batch_size, seq_length):
    train_ds = WeatherDataset(train_set, seq_length)
    val_ds = WeatherDataset(val_set, seq_length)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl
