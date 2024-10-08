from doc import Dataset
train_dataset = Dataset(path='data/WN18RR/train.txt.json', task='WN18RR')
valid_dataset = Dataset(path='data/WN18RR/valid.txt.json', task='WN18RR')
test_dataset = Dataset(path='data/WN18RR/test.txt.json', task='WN18RR')