import torch
import torch.nn as nn
import math
import numpy as np

'''
上下文感知   词嵌入层转化向量  位置编码  多头注意力
上下文深度   非线性FFN
最终分类     残差连接+归一化   聚合+线性分类头
'''
class PositionEncoding(nn.Module):
    '''
    位置编码层：为模型注入序列的位置信息
    在相关联性中 点积并不处理位置信息，需要有一个位置表示方案
    唯一性 确定性 泛化性 表达相对位置
    '''
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=float).unsqueeze(1) #（5000，1）
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) #(d_model/2)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        self.register_buffer('pe',pe.unsqueeze(0)) #buffer不是参数不会被更改，（1，5000，d_model)

    def forward(self,x):
        #将位置编码添加到词嵌入上
        x = x + self.pe[:,:x.size(1),:] #区分整数索引和切片索引
        return x
    
class MultiHeadAttention(nn.Module):
    '''
    注意力层： 上下文感知，多头注意力 input:(batch_size,seq_len,d_model)
    '''
    def __init__(self, num_head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_head ==0,"d_model must be divisible by numhead"
        self.numhead = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
    
    def ScaledDotProductAttention(self,query,key,value,mask=None):
        '''
        点积求相关系数，自注意力
        '''
        #(batch_size,num_head,seq_len,d_k)  mask:(batch_size,seq_len)
        scores = torch.matmul(query,key.transpose(2,3)) / torch.sqrt(torch.tensor(key.size(-1)))
        if mask is not None:
            #(,,seq,seq) 
            scores = scores.masked_fill(mask==0,-1e9)
        attention_weight = torch.softmax(scores,dim=-1)
        attention_weight = self.dropout(attention_weight)
        output = torch.matmul(attention_weight,value) #(,,seq,d_k)
        return output
    
    def forward(self,query,key,value,mask=None):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        #分头(,num_head,,d_k)
        q = q.view(query.size(0),-1,self.numhead,self.d_k).transpose(1,2)
        k = k.view(key.size(0),-1,self.numhead,self.d_k).transpose(1,2)
        v = v.view(value.size(0),-1,self.numhead,self.d_k).transpose(1,2)

        if mask is not None:
            #(batch_size,seq_len)->(,1,1,)
            mask = mask.unsqueeze(1).unsqueeze(1)

        context = self.ScaledDotProductAttention(q,k,v,mask)
        context = context.transpose(1,2).contiguous().view(query.size(0),-1,self.d_model)
        #转换回(batch_size,seq_len,d_model)才可以用W_o线性层转化
        output = self.W_o(context)
        return output 

class PositionWiseFeedForward(nn.Module):
    '''
    FFN  非线性能力 input:(batch_size,seq,d_model) output:(batch_size,seq_len,d_model)
    '''
    def __init__(self, d_model, d_ff,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(self.dropout(x))
        return x
    
class EncoderLayer(nn.Module):
    '''
    单一的encoder层
    '''
    def __init__(self, d_model, d_ff, num_head, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_head,d_model,dropout)
        self.ffn = PositionWiseFeedForward(d_model,d_ff,dropout)
        self.nom1 = nn.LayerNorm(d_model) #特征归一化 在d_model维度上
        self.nom2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attn_output = self.attn(x,x,x,mask)
        x = self.nom1(x + self.drop1(attn_output)) #x + 残差 防梯度消失
        ffn_output = self.ffn(x)
        x = self.nom2(x + self.drop2(ffn_output))
        return x

class transformerBlock(nn.Module):
    def __init__(self, num_classes, num_layers, d_model, d_ff, num_head, vocab_len, dropout=0.1):
        super().__init__()
        #1.嵌入层
        self.embedding = nn.Embedding(vocab_len,d_model)
        #2.位置编码层
        self.pos_encoder = PositionEncoding(d_model)
        #3.encoder层
        self.encoder = nn.ModuleList([EncoderLayer(d_model,d_ff,num_head) for _ in range(num_layers)])
        #4.分类头
        self.classifier = nn.Linear(d_model,num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,src_mask=None):
        #1.词嵌入 (batch_size,seq,d_model)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        #2.通过num_layers个encoder
        for layer in self.encoder:
            src = layer(src,src_mask)
        #3.分类  简单求平均 (batch_size,seq_len,d_model)->(batch_size,d_model)
        pooled_output = src.mean(dim=1)
        logits = self.classifier(self.dropout(pooled_output)) #logits还不是概率(softmax)

        return logits




