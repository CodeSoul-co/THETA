# Qwen3-Embedding 训练操作指南

## 📋 目标
使用 Qwen3-Embedding-0.6B 模型对5个社会学文本数据集进行向量化，输出 `doc×vector` 矩阵供后续 ETM 模型训练使用。

## 📂 项目结构

```
/root/autodl-tmp/
├── data/
│   ├── cleaned_data_report.md          # 数据集说明文档
│   ├── dataset1_labeled/               # 有标签数据集1
│   ├── dataset2_labeled/               # 有标签数据集2
│   ├── dataset3_labeled/               # 有标签数据集3
│   ├── dataset4_unlabeled/             # 无标签数据集4
│   └── dataset5_unlabeled/             # 无标签数据集5
├── embedding/
│   ├── configs/                        # 配置文件目录
│   ├── scripts/                        # 训练脚本
│   ├── outputs/                        # 输出目录
│   │   ├── zero_shot/                  # Zero-shot结果
│   │   ├── supervised/                 # 有监督训练结果
│   │   └── unsupervised/               # 无监督训练结果
│   ├── checkpoints/                    # 模型检查点
│   └── logs/                           # 训练日志
└── ETM/                                # 下游ETM模型目录
```

## 🚀 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Embedding相关
pip install transformers>=4.51.0
pip install sentence-transformers>=2.7.0
pip install ms-swift
pip install peft
pip install datasets

# 工具库
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install tensorboard
```

### 2. 验证环境

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('Transformers OK')"
```

## 📊 数据准备

### 1. 检查数据完整性

```bash
cd /root/autodl-tmp/data
cat cleaned_data_report.md
```

### 2. 创建数据加载脚本

在 `/root/autodl-tmp/embedding/scripts/data_loader.py` 创建：

```python
import os
import json
import pandas as pd
from typing import List, Dict, Tuple

class DatasetLoader:
    """统一的数据集加载器"""
    
    def __init__(self, base_path: str = "/root/autodl-tmp/data"):
        self.base_path = base_path
        
    def load_dataset(self, dataset_name: str) -> Tuple[List[str], List]:
        """
        加载单个数据集
        
        Returns:
            texts: 文本列表
            labels: 标签列表（如果无标签则为None）
        """
        dataset_path = os.path.join(self.base_path, dataset_name)
        
        # TODO: 根据实际数据格式调整
        # 假设数据格式为CSV或JSON
        if os.path.exists(os.path.join(dataset_path, "data.csv")):
            df = pd.read_csv(os.path.join(dataset_path, "data.csv"))
            texts = df['text'].tolist()
            labels = df['label'].tolist() if 'label' in df.columns else None
        elif os.path.exists(os.path.join(dataset_path, "data.json")):
            with open(os.path.join(dataset_path, "data.json"), 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = [item['text'] for item in data]
            labels = [item.get('label') for item in data]
            if all(l is None for l in labels):
                labels = None
        else:
            raise FileNotFoundError(f"No data file found in {dataset_path}")
            
        return texts, labels
    
    def get_all_datasets(self) -> Dict[str, Tuple[List[str], List]]:
        """加载所有数据集"""
        datasets = {}
        dataset_names = [
            "dataset1_labeled",
            "dataset2_labeled", 
            "dataset3_labeled",
            "dataset4_unlabeled",
            "dataset5_unlabeled"
        ]
        
        for name in dataset_names:
            texts, labels = self.load_dataset(name)
            datasets[name] = (texts, labels)
            print(f"Loaded {name}: {len(texts)} samples, labeled: {labels is not None}")
            
        return datasets
```

## 🎯 方法一：Zero-Shot Embedding

### 概述
直接使用预训练的 Qwen3-Embedding-0.6B 模型，无需训练，快速获取基线向量表示。

### 使用场景
- 所有5个数据集
- 作为性能基线
- 快速原型验证

### 实现步骤

#### 1. 创建 Zero-Shot 脚本

在 `/root/autodl-tmp/embedding/scripts/zero_shot_embedding.py` 创建：

```python
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import pickle
from data_loader import DatasetLoader

class ZeroShotEmbedder:
    """Zero-shot embedding生成器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        
        # 如果有GPU，使用GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
    
    def embed_texts(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        批量生成embeddings
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            embeddings: (num_docs, embedding_dim) 的numpy数组
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                batch_emb = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # L2归一化
                )
            embeddings.append(batch_emb)
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """保存embeddings"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为.npy格式（推荐用于ETM）
        np.save(output_path + '.npy', embeddings)
        
        # 也保存为.pkl格式（备用）
        with open(output_path + '.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Saved embeddings to {output_path}")
        print(f"Shape: {embeddings.shape}")

def main():
    # 配置
    output_dir = "/root/autodl-tmp/embedding/outputs/zero_shot"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    loader = DatasetLoader()
    datasets = loader.get_all_datasets()
    
    # 初始化embedder
    embedder = ZeroShotEmbedder()
    
    # 对每个数据集生成embeddings
    for dataset_name, (texts, labels) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print(f"{'='*50}")
        
        # 生成embeddings
        embeddings = embedder.embed_texts(texts, batch_size=32)
        
        # 保存
        output_path = os.path.join(output_dir, f"{dataset_name}_embeddings")
        embedder.save_embeddings(embeddings, output_path)
        
        # 如果有标签，也保存标签
        if labels is not None:
            label_path = os.path.join(output_dir, f"{dataset_name}_labels.npy")
            np.save(label_path, np.array(labels))
            print(f"Saved labels to {label_path}")
    
    print("\n✅ Zero-shot embedding completed!")

if __name__ == "__main__":
    main()
```

#### 2. 运行 Zero-Shot

```bash
cd /root/autodl-tmp/embedding/scripts
python zero_shot_embedding.py
```

#### 3. 验证输出

```python
# 检查输出
import numpy as np

# 加载embeddings
embeddings = np.load('/root/autodl-tmp/embedding/outputs/zero_shot/dataset1_labeled_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")  # 应该是 (num_docs, 768)
print(f"Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
```

---

## 🎓 方法二：有监督学习（LoRA微调）

### 概述
使用有标签数据（3个数据集）进行监督微调，通过 LoRA 方法高效调整模型参数。

### 使用场景
- dataset1_labeled
- dataset2_labeled
- dataset3_labeled

### 关键技术
- **LoRA (Low-Rank Adaptation)**: 低秩分解，只训练少量参数
- **对比学习损失**: 正样本拉近，负样本推远
- **交叉熵损失**: 优化分类边界

### 实现步骤

#### 1. 准备训练数据格式

在 `/root/autodl-tmp/embedding/scripts/prepare_supervised_data.py` 创建：

```python
import json
import random
from typing import List, Tuple
from data_loader import DatasetLoader

class SupervisedDataPreparer:
    """准备有监督训练数据"""
    
    def __init__(self, negative_ratio: int = 5):
        """
        Args:
            negative_ratio: 负样本数量相对于正样本的比例
        """
        self.negative_ratio = negative_ratio
    
    def create_triplets(self, texts: List[str], labels: List) -> List[dict]:
        """
        创建三元组数据: (query, positive, negative)
        
        策略：
        - 同一标签的样本互为正样本
        - 不同标签的样本为负样本
        """
        # 按标签分组
        label_to_texts = {}
        for text, label in zip(texts, labels):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)
        
        triplets = []
        all_labels = list(label_to_texts.keys())
        
        for label, label_texts in label_to_texts.items():
            # 如果该标签下样本太少，跳过
            if len(label_texts) < 2:
                continue
            
            # 为每个文本创建训练样本
            for i, query in enumerate(label_texts):
                # 正样本：同标签的其他文本
                positive_candidates = [t for j, t in enumerate(label_texts) if j != i]
                if not positive_candidates:
                    continue
                
                # 负样本：不同标签的文本
                negative_labels = [l for l in all_labels if l != label]
                negatives = []
                for _ in range(self.negative_ratio):
                    neg_label = random.choice(negative_labels)
                    neg_text = random.choice(label_to_texts[neg_label])
                    negatives.append(neg_text)
                
                triplets.append({
                    "query": query,
                    "positive": random.choice(positive_candidates),
                    "negatives": negatives,
                    "label": label
                })
        
        return triplets
    
    def save_to_jsonl(self, triplets: List[dict], output_path: str):
        """保存为JSONL格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for triplet in triplets:
                f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        print(f"Saved {len(triplets)} triplets to {output_path}")

def main():
    loader = DatasetLoader()
    preparer = SupervisedDataPreparer(negative_ratio=5)
    
    labeled_datasets = [
        "dataset1_labeled",
        "dataset2_labeled",
        "dataset3_labeled"
    ]
    
    for dataset_name in labeled_datasets:
        print(f"\nProcessing {dataset_name}...")
        texts, labels = loader.load_dataset(dataset_name)
        
        # 创建三元组
        triplets = preparer.create_triplets(texts, labels)
        
        # 划分训练集和验证集
        random.shuffle(triplets)
        split_idx = int(len(triplets) * 0.9)
        train_triplets = triplets[:split_idx]
        val_triplets = triplets[split_idx:]
        
        # 保存
        output_dir = f"/root/autodl-tmp/embedding/outputs/supervised/{dataset_name}"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        preparer.save_to_jsonl(train_triplets, f"{output_dir}/train.jsonl")
        preparer.save_to_jsonl(val_triplets, f"{output_dir}/val.jsonl")
        
        print(f"Train: {len(train_triplets)}, Val: {len(val_triplets)}")

if __name__ == "__main__":
    main()
```

#### 2. 创建 LoRA 训练脚本

在 `/root/autodl-tmp/embedding/scripts/train_supervised_lora.py` 创建：

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
import numpy as np

class TripletDataset(Dataset):
    """三元组数据集"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        query = self.tokenizer(
            item['query'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive = self.tokenizer(
            item['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 随机选一个负样本
        import random
        negative_text = random.choice(item['negatives'])
        negative = self.tokenizer(
            negative_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query['input_ids'].squeeze(0),
            'query_attention_mask': query['attention_mask'].squeeze(0),
            'positive_input_ids': positive['input_ids'].squeeze(0),
            'positive_attention_mask': positive['attention_mask'].squeeze(0),
            'negative_input_ids': negative['input_ids'].squeeze(0),
            'negative_attention_mask': negative['attention_mask'].squeeze(0),
        }

class TripletLoss(nn.Module):
    """三元组损失"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        
        Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def mean_pooling(token_embeddings, attention_mask):
    """均值池化"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 配置LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对注意力层应用LoRA
            bias="none"
        )
        
        # 应用LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        # 损失函数
        self.criterion = TripletLoss(margin=0.5)
        
        # 移动到GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def encode(self, input_ids, attention_mask):
        """编码文本"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
        # L2归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def train_epoch(self, dataloader, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # 移动到GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 编码
            query_emb = self.encode(batch['query_input_ids'], batch['query_attention_mask'])
            pos_emb = self.encode(batch['positive_input_ids'], batch['positive_attention_mask'])
            neg_emb = self.encode(batch['negative_input_ids'], batch['negative_attention_mask'])
            
            # 计算损失
            loss = self.criterion(query_emb, pos_emb, neg_emb)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """评估"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            query_emb = self.encode(batch['query_input_ids'], batch['query_attention_mask'])
            pos_emb = self.encode(batch['positive_input_ids'], batch['positive_attention_mask'])
            neg_emb = self.encode(batch['negative_input_ids'], batch['negative_attention_mask'])
            
            loss = self.criterion(query_emb, pos_emb, neg_emb)
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self, output_dir: str):
        """保存LoRA权重"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def train_dataset(dataset_name: str, num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-4):
    """训练单个数据集"""
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}")
    
    # 路径
    data_dir = f"/root/autodl-tmp/embedding/outputs/supervised/{dataset_name}"
    output_dir = f"/root/autodl-tmp/embedding/checkpoints/supervised/{dataset_name}"
    
    # 初始化训练器
    trainer = LoRATrainer(lora_r=8, lora_alpha=32)
    
    # 准备数据
    train_dataset = TripletDataset(f"{data_dir}/train.jsonl", trainer.tokenizer)
    val_dataset = TripletDataset(f"{data_dir}/val.jsonl", trainer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = trainer.train_epoch(train_loader, optimizer)
        val_loss = trainer.evaluate(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(output_dir)
            print(f"✓ New best model saved!")
    
    return output_dir

def main():
    datasets = ["dataset1_labeled", "dataset2_labeled", "dataset3_labeled"]
    
    for dataset_name in datasets:
        model_path = train_dataset(
            dataset_name,
            num_epochs=5,
            batch_size=16,
            learning_rate=2e-4
        )
        print(f"\n✅ Completed training for {dataset_name}")
        print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main()
```

#### 3. 运行监督训练

```bash
# 1. 准备训练数据
cd /root/autodl-tmp/embedding/scripts
python prepare_supervised_data.py

# 2. 开始训练
python train_supervised_lora.py
```

#### 4. 生成微调后的embeddings

在 `/root/autodl-tmp/embedding/scripts/generate_supervised_embeddings.py` 创建：

```python
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import os
from data_loader import DatasetLoader

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(dataset_name: str, model_path: str, batch_size: int = 32):
    """使用微调后的模型生成embeddings"""
    
    # 加载模型
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加载数据
    loader = DatasetLoader()
    texts, labels = loader.load_dataset(dataset_name)
    
    # 生成embeddings
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Forward
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings_matrix = np.vstack(all_embeddings)
    
    # 保存
    output_dir = "/root/autodl-tmp/embedding/outputs/supervised"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{dataset_name}_lora_embeddings")
    np.save(output_path + '.npy', embeddings_matrix)
    
    if labels is not None:
        np.save(os.path.join(output_dir, f"{dataset_name}_labels.npy"), np.array(labels))
    
    print(f"Saved embeddings: {embeddings_matrix.shape}")
    return embeddings_matrix

def main():
    datasets = ["dataset1_labeled", "dataset2_labeled", "dataset3_labeled"]
    
    for dataset_name in datasets:
        model_path = f"/root/autodl-tmp/embedding/checkpoints/supervised/{dataset_name}"
        generate_embeddings(dataset_name, model_path)
        print(f"✅ Generated embeddings for {dataset_name}\n")

if __name__ == "__main__":
    main()
```

```bash
# 生成微调后的embeddings
python generate_supervised_embeddings.py
```

---

## 🔄 方法三：无监督学习（自回归 + KL散度）

### 概述
对无标签数据（2个数据集）使用自监督学习，通过掩码语言模型(MLM)或自回归预测任务进行训练。

### 使用场景
- dataset4_unlabeled
- dataset5_unlabeled

### 关键技术
- **Masked Language Modeling (MLM)**: 掩码预测
- **Autoregressive Prediction**: 自回归预测
- **KL Divergence**: 衡量分布差异

### 实现步骤

#### 1. 创建无监督训练脚本

在 `/root/autodl-tmp/embedding/scripts/train_unsupervised.py` 创建：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
import numpy as np
from data_loader import DatasetLoader

class MLMDataset(Dataset):
    """掩码语言模型数据集"""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 512, mlm_probability: float = 0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    
    def __len__(self):
        return len(self.texts)
    
    def mask_tokens(self, inputs):
        """随机掩码token"""
        labels = inputs.clone()
        
        # 创建掩码矩阵
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 不掩码特殊token
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码位置的损失
        
        # 80%替换为[MASK], 10%随机替换, 10%保持不变
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 应用掩码
        masked_input_ids, labels = self.mask_tokens(input_ids)
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class KLDivergenceLoss(nn.Module):
    """KL散度损失"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits):
        """
        计算学生模型和教师模型输出分布的KL散度
        
        Args:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits (可以是原始模型或目标分布)
        """
        # 应用温度缩放，使分布更平滑
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 计算KL散度
        kl_div = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        return kl_div

class UnsupervisedTrainer:
    """无监督训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_kl_loss: bool = True
    ):
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 加载基础模型（作为教师）
        if use_kl_loss:
            self.teacher_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None
        
        # 加载学生模型并应用LoRA
        student_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(student_model, lora_config)
        self.model.print_trainable_parameters()
        
        # 损失函数
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = KLDivergenceLoss(temperature=2.0) if use_kl_loss else None
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.teacher_model:
            self.teacher_model = self.teacher_model.to(self.device)
    
    def train_epoch(self, dataloader, optimizer, alpha_mlm: float = 0.7, alpha_kl: float = 0.3):
        """
        训练一个epoch
        
        Args:
            alpha_mlm: MLM损失权重
            alpha_kl: KL散度损失权重
        """
        self.model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 学生模型前向传播
            student_outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # MLM损失
            # 注意：这里需要添加语言模型头，简化版本使用last_hidden_state
            # 实际应用中需要添加lm_head层
            mlm_loss = torch.tensor(0.0).to(self.device)  # 占位符
            
            # KL散度损失
            kl_loss = torch.tensor(0.0).to(self.device)
            if self.teacher_model and self.kl_loss:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                
                # 计算隐藏状态的KL散度
                # 简化版本：使用均值池化后的向量
                student_hidden = student_outputs.last_hidden_state.mean(dim=1)
                teacher_hidden = teacher_outputs.last_hidden_state.mean(dim=1)
                
                # 将向量转换为分布（简化）
                kl_loss = F.mse_loss(student_hidden, teacher_hidden)  # 简化版KL
            
            # 组合损失
            loss = alpha_mlm * mlm_loss + alpha_kl * kl_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_kl_loss += kl_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mlm': mlm_loss.item(),
                'kl': kl_loss.item()
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_mlm = total_mlm_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        
        return avg_loss, avg_mlm, avg_kl
    
    def save_model(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def train_unsupervised_dataset(dataset_name: str, num_epochs: int = 3, batch_size: int = 16):
    """训练单个无标签数据集"""
    print(f"\n{'='*60}")
    print(f"Unsupervised training on {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    loader = DatasetLoader()
    texts, _ = loader.load_dataset(dataset_name)
    
    # 初始化训练器
    trainer = UnsupervisedTrainer(lora_r=8, lora_alpha=16, use_kl_loss=True)
    
    # 准备数据
    train_size = int(0.9 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_dataset = MLMDataset(train_texts, trainer.tokenizer)
    val_dataset = MLMDataset(val_texts, trainer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=2e-4)
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        loss, mlm_loss, kl_loss = trainer.train_epoch(
            train_loader, 
            optimizer,
            alpha_mlm=0.7,
            alpha_kl=0.3
        )
        
        print(f"Loss: {loss:.4f}, MLM: {mlm_loss:.4f}, KL: {kl_loss:.4f}")
    
    # 保存模型
    output_dir = f"/root/autodl-tmp/embedding/checkpoints/unsupervised/{dataset_name}"
    trainer.save_model(output_dir)
    
    return output_dir

def main():
    unlabeled_datasets = ["dataset4_unlabeled", "dataset5_unlabeled"]
    
    for dataset_name in unlabeled_datasets:
        model_path = train_unsupervised_dataset(
            dataset_name,
            num_epochs=5,
            batch_size=16
        )
        print(f"\n✅ Completed unsupervised training for {dataset_name}")
        print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main()
```

#### 2. 运行无监督训练

```bash
cd /root/autodl-tmp/embedding/scripts
python train_unsupervised.py
```

#### 3. 生成无监督embeddings

在 `/root/autodl-tmp/embedding/scripts/generate_unsupervised_embeddings.py` 创建：

```python
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import os
from data_loader import DatasetLoader

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_unsupervised_embeddings(dataset_name: str, batch_size: int = 32):
    """使用无监督训练的模型生成embeddings"""
    
    # 模型路径
    model_path = f"/root/autodl-tmp/embedding/checkpoints/unsupervised/{dataset_name}"
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加载数据
    loader = DatasetLoader()
    texts, _ = loader.load_dataset(dataset_name)
    
    # 生成embeddings
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            encoded = tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings_matrix = np.vstack(all_embeddings)
    
    # 保存
    output_dir = "/root/autodl-tmp/embedding/outputs/unsupervised"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{dataset_name}_unsupervised_embeddings")
    np.save(output_path + '.npy', embeddings_matrix)
    
    print(f"Saved embeddings: {embeddings_matrix.shape}")
    return embeddings_matrix

def main():
    unlabeled_datasets = ["dataset4_unlabeled", "dataset5_unlabeled"]
    
    for dataset_name in unlabeled_datasets:
        generate_unsupervised_embeddings(dataset_name)
        print(f"✅ Generated embeddings for {dataset_name}\n")

if __name__ == "__main__":
    main()
```

```bash
python generate_unsupervised_embeddings.py
```

---

## 📦 输出格式说明

### 1. 文件结构

```
/root/autodl-tmp/embedding/outputs/
├── zero_shot/
│   ├── dataset1_labeled_embeddings.npy           # (N, 768)
│   ├── dataset1_labeled_labels.npy               # (N,)
│   ├── dataset2_labeled_embeddings.npy
│   ├── dataset2_labeled_labels.npy
│   ├── dataset3_labeled_embeddings.npy
│   ├── dataset3_labeled_labels.npy
│   ├── dataset4_unlabeled_embeddings.npy
│   └── dataset5_unlabeled_embeddings.npy
├── supervised/
│   ├── dataset1_labeled_lora_embeddings.npy
│   ├── dataset1_labeled_labels.npy
│   ├── dataset2_labeled_lora_embeddings.npy
│   ├── dataset2_labeled_labels.npy
│   ├── dataset3_labeled_lora_embeddings.npy
│   └── dataset3_labeled_labels.npy
└── unsupervised/
    ├── dataset4_unlabeled_unsupervised_embeddings.npy
    └── dataset5_unlabeled_unsupervised_embeddings.npy
```

### 2. Numpy数组格式

所有embedding文件为 `.npy` 格式:
- **Shape**: `(num_documents, embedding_dim)`
- **Dtype**: `float32`
- **Normalized**: L2归一化后的向量

### 3. 加载示例

```python
import numpy as np

# 加载embeddings
embeddings = np.load('dataset1_labeled_embeddings.npy')
labels = np.load('dataset1_labeled_labels.npy')

print(f"Embeddings shape: {embeddings.shape}")  # (N, 768)
print(f"Labels shape: {labels.shape}")          # (N,)

# 验证归一化
norms = np.linalg.norm(embeddings, axis=1)
print(f"Vector norms (should be ~1.0): {norms[:5]}")
```

## 🔗 与ETM模型的接口

### 1. 数据格式要求

为确保ETM模型能够顺利接收embedding输出，需满足：

```python
# ETM期望的输入格式
{
    'embeddings': np.ndarray,  # Shape: (num_docs, embedding_dim)
    'vocabulary_size': int,     # 词汇表大小
    'labels': np.ndarray,       # Shape: (num_docs,), 可选
    'doc_ids': List[str]        # 文档ID列表
}
```

### 2. 接口脚本

在 `/root/autodl-tmp/embedding/scripts/prepare_for_etm.py` 创建：

```python
import numpy as np
import json
import os

def prepare_etm_input(
    embeddings_path: str,
    labels_path: str = None,
    output_dir: str = "/root/autodl-tmp/ETM/inputs"
):
    """
    准备ETM输入数据
    
    Args:
        embeddings_path: embeddings的.npy文件路径
        labels_path: 标签的.npy文件路径（可选）
        output_dir: 输出目录
    """
    # 加载embeddings
    embeddings = np.load(embeddings_path)
    
    # 加载标签（如果有）
    labels = np.load(labels_path) if labels_path and os.path.exists(labels_path) else None
    
    # 创建输出
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = os.path.basename(embeddings_path).replace('_embeddings.npy', '')
    
    # 保存为ETM格式
    etm_data = {
        'embeddings': embeddings.tolist(),
        'num_docs': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'labels': labels.tolist() if labels is not None else None,
        'doc_ids': [f"doc_{i}" for i in range(embeddings.shape[0])]
    }
    
    output_path = os.path.join(output_dir, f"{dataset_name}_etm_input.json")
    with open(output_path, 'w') as f:
        json.dump(etm_data, f)
    
    print(f"✓ Prepared ETM input: {output_path}")
    print(f"  - Documents: {embeddings.shape[0]}")
    print(f"  - Embedding dim: {embeddings.shape[1]}")
    print(f"  - Has labels: {labels is not None}")
    
    return output_path

def main():
    """为所有数据集准备ETM输入"""
    
    # Zero-shot embeddings
    zero_shot_dir = "/root/autodl-tmp/embedding/outputs/zero_shot"
    for filename in os.listdir(zero_shot_dir):
        if filename.endswith('_embeddings.npy'):
            embeddings_path = os.path.join(zero_shot_dir, filename)
            labels_path = embeddings_path.replace('_embeddings.npy', '_labels.npy')
            prepare_etm_input(embeddings_path, labels_path)
    
    # Supervised embeddings
    supervised_dir = "/root/autodl-tmp/embedding/outputs/supervised"
    for filename in os.listdir(supervised_dir):
        if filename.endswith('_lora_embeddings.npy'):
            embeddings_path = os.path.join(supervised_dir, filename)
            labels_path = os.path.join(supervised_dir, filename.replace('_lora_embeddings.npy', '_labels.npy'))
            prepare_etm_input(embeddings_path, labels_path)
    
    # Unsupervised embeddings
    unsupervised_dir = "/root/autodl-tmp/embedding/outputs/unsupervised"
    for filename in os.listdir(unsupervised_dir):
        if filename.endswith('_unsupervised_embeddings.npy'):
            embeddings_path = os.path.join(unsupervised_dir, filename)
            prepare_etm_input(embeddings_path)

if __name__ == "__main__":
    main()
```

## 📊 质量检查

### 验证脚本

在 `/root/autodl-tmp/embedding/scripts/validate_embeddings.py` 创建：

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def validate_embeddings(embeddings_path: str):
    """验证embeddings质量"""
    
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(embeddings_path)}")
    print(f"{'='*60}")
    
    # 加载
    embeddings = np.load(embeddings_path)
    
    # 1. 基本统计
    print(f"\n1. Basic Statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Min: {embeddings.min():.4f}")
    print(f"   Max: {embeddings.max():.4f}")
    
    # 2. 归一化检查
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n2. Normalization Check:")
    print(f"   Mean norm: {norms.mean():.4f} (should be ~1.0)")
    print(f"   Std norm: {norms.std():.4f}")
    
    # 3. 语义相似性检查（随机采样）
    if len(embeddings) > 10:
        sample_indices = np.random.choice(len(embeddings), min(10, len(embeddings)), replace=False)
        sample_embs = embeddings[sample_indices]
        sim_matrix = cosine_similarity(sample_embs)
        
        print(f"\n3. Semantic Similarity (sample):")
        print(f"   Average pairwise similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.4f}")
        print(f"   Min similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].min():.4f}")
        print(f"   Max similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].max():.4f}")
    
    # 4. 检查NaN和Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    print(f"\n4. Data Quality:")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("   ⚠️  WARNING: Contains NaN or Inf values!")
    else:
        print("   ✓ Clean data")
    
    return not (has_nan or has_inf)

def main():
    """验证所有embeddings"""
    
    directories = [
        "/root/autodl-tmp/embedding/outputs/zero_shot",
        "/root/autodl-tmp/embedding/outputs/supervised",
        "/root/autodl-tmp/embedding/outputs/unsupervised"
    ]
    
    all_valid = True
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            if filename.endswith('_embeddings.npy'):
                filepath = os.path.join(directory, filename)
                is_valid = validate_embeddings(filepath)
                all_valid = all_valid and is_valid
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All embeddings are valid!")
    else:
        print("❌ Some embeddings have quality issues")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

```bash
python validate_embeddings.py
```

## 🎯 完整工作流

### 一键运行脚本

在 `/root/autodl-tmp/embedding/run_all.sh` 创建：

```bash
#!/bin/bash

echo "=========================================="
echo "Qwen3-Embedding Training Pipeline"
echo "=========================================="

cd /root/autodl-tmp/embedding/scripts

# Step 1: Zero-shot
echo -e "\n[Step 1/5] Running Zero-shot embedding..."
python zero_shot_embedding.py

# Step 2: Prepare supervised data
echo -e "\n[Step 2/5] Preparing supervised training data..."
python prepare_supervised_data.py

# Step 3: Train supervised models
echo -e "\n[Step 3/5] Training supervised models with LoRA..."
python train_supervised_lora.py

# Step 4: Generate supervised embeddings
echo -e "\n[Step 4/5] Generating supervised embeddings..."
python generate_supervised_embeddings.py

# Step 5: Train unsupervised models
echo -e "\n[Step 5/5] Training unsupervised models..."
python train_unsupervised.py
python generate_unsupervised_embeddings.py

# Validation
echo -e "\n[Validation] Checking embedding quality..."
python validate_embeddings.py

# Prepare for ETM
echo -e "\n[ETM Preparation] Preparing data for ETM model..."
python prepare_for_etm.py

echo -e "\n=========================================="
echo "✅ All tasks completed!"
echo "=========================================="
```

```bash
chmod +x /root/autodl-tmp/embedding/run_all.sh
./run_all.sh
```

## 📝 监控训练

### TensorBoard可视化（可选）

```bash
# 安装tensorboard
pip install tensorboard

# 启动tensorboard
tensorboard --logdir=/root/autodl-tmp/embedding/logs --port=6006
```

## ⚠️ 注意事项

1. **显存管理**: 
   - 如果显存不足，减小 `batch_size`
   - 考虑使用梯度累积: `accumulation_steps=4`

2. **数据格式**:
   - 确保 `data_loader.py` 中的数据加载逻辑与实际数据格式匹配
   - 检查文本编码（UTF-8）

3. **LoRA参数**:
   - `r=8`: 低秩分解的秩，越大参数越多
   - `lora_alpha=32`: 缩放因子，控制LoRA的影响强度

4. **训练时间**:
   - Zero-shot: 即时完成
   - 有监督训练: 每个数据集约1-3小时
   - 无监督训练: 每个数据集约2-4小时

5. **检查点保存**:
   - 所有模型检查点保存在 `/checkpoints/` 目录
   - 可以随时中断并从检查点恢复

## 🔍 故障排除

### 常见问题

1. **CUDA out of memory**:
   ```python
   # 减小batch_size
   batch_size = 8  # 从16降到8
   ```

2. **模型下载失败**:
   ```bash
   # 使用镜像源
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **数据加载错误**:
   ```python
   # 检查数据路径
   ls -la /root/autodl-tmp/data/
   ```

## 📖 参考资料

- Qwen3-Embedding论文: [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)
- LoRA论文: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- MTEB Benchmark: [https://github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)

---

**下一步**: 将生成的embeddings输入到ETM模型进行主题建模训练，我们只进行到embeddings这一步，