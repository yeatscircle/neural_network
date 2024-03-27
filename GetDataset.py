import os
import pandas
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Download with: python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, senquences_list):
        frequencies = {}
        id = 4

        for sequence in senquences_list:
            for word in self.tokenizer_eng(sequence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == 5:
                    self.stoi[word] = id
                    self.itos[id] = word
                    id += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class MyDataset(Dataset):
    def __init__(self, filename, root_dir, transformers=None, freq_threshold=5):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.df = pandas.read_csv(filename)
        self.transformers = transformers

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.open(os.path.join(self.root_dir, img)).convert("RGB")
        if self.transformers is not None:
            img = self.transformers(img)

        temp = self.vocab.stoi['<SOS>']
        numericalized_caption = self.vocab.numericalize(self.captions[idx])
        numericalized_caption.insert(0, temp)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # 扩展一个维度(1,channel,height,width)
        imgs = [item[0].unsqueezze(0) for item in batch]
        # 把多个拼接起来(batch,channel,height,width)
        imgs = torch.cat(imgs, dim=0)
        # 以长度作为input的标准,也就是embedding vector转换为同一个维度
        targets = [item[1] for item in batch]
        # 这个位置会使得原来的横向向量转置
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    my_dataset = MyDataset(annotation_file, root_folder, transform)
    pad_idx = my_dataset.vocab.stoi['<PAD>']

    my_dataLoader = DataLoader(
        dataset=my_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return my_dataset, my_dataLoader


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    loader, dataset = get_loader(
        "flickr8k/Images", "flickr8k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
