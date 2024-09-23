from transformers import AutoModel, AutoTokenizer, AutoConfig

# 加载配置
config = AutoConfig.from_pretrained('/mnt/data/yhy/model/deberta-v3-base')

# 从 `pytorch_model.bin` 加载模型
model = AutoModel.from_pretrained('/mnt/data/yhy/model/deberta-v3-base', config=config)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/yhy/model/deberta-v3-base')

# 保存为 Hugging Face 格式
model.save_pretrained('/mnt/data/yhy/model/-hf')
tokenizer.save_pretrained('/mnt/data/yhy/model/deberta-v3-base-hf')
model = AutoModel.from_pretrained('/mnt/data/yhy/model/deberta-v3-base-hf')
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/yhy/model/deberta-v3-base-hf')
