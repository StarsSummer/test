#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于BERT架构的模型训练脚本
支持训练集和验证集的加载、训练和验证
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
import logging
from tqdm import tqdm
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    """基于BERT的分类器"""
    def __init__(self, n_classes, pretrained_model_name='bert-base-chinese'):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

class BertTrainer:
    """BERT训练器"""
    def __init__(self, model, train_loader, val_loader, device, n_classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_classes = n_classes
        
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, learning_rate=2e-5, epochs=3):
        """设置优化器和学习率调度器"""
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        total_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch in tqdm(self.train_loader, desc="训练中"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
        
        return total_loss / len(self.train_loader), correct_predictions.double() / len(self.train_loader.dataset)
    
    def eval_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
        
        return total_loss / len(self.val_loader), correct_predictions.double() / len(self.val_loader.dataset)
    
    def train(self, epochs=3, save_path='bert_model.pth'):
        """完整训练流程"""
        best_accuracy = 0
        
        for epoch in range(epochs):
            logger.info(f'Epoch {epoch + 1}/{epochs}')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            
            logger.info(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
            logger.info(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(f'保存最佳模型到 {save_path}')

def load_data(train_path, val_path=None, test_size=0.2):
    """加载数据"""
    # 假设数据格式为CSV，包含'text'和'label'列
    if val_path is None:
        # 如果没有提供验证集，从训练集中分割
        df = pd.read_csv(train_path)
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    
    return train_df, val_df

def main():
    parser = argparse.ArgumentParser(description='BERT模型训练脚本')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_data', type=str, help='验证数据路径')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese', help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--save_path', type=str, default='bert_model.pth', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 加载数据
    train_df, val_df = load_data(args.train_data, args.val_data)
    logger.info(f'训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}')
    
    # 获取类别数量
    n_classes = train_df['label'].nunique()
    logger.info(f'类别数量: {n_classes}')
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # 创建数据集
    train_dataset = TextDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = TextDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化模型
    model = BertClassifier(n_classes=n_classes, pretrained_model_name=args.model_name)
    model = model.to(device)
    
    # 初始化训练器
    trainer = BertTrainer(model, train_loader, val_loader, device, n_classes)
    trainer.setup_optimizer(learning_rate=args.learning_rate, epochs=args.epochs)
    
    # 开始训练
    trainer.train(epochs=args.epochs, save_path=args.save_path)
    
    logger.info('训练完成!')

if __name__ == '__main__':
    main()