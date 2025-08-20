import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import transformerBlock
import numpy as np
import matplotlib.pyplot as plt
import json

with open('config.json','r') as f:
    config = json.load(f)

#toyset 1:positive 0:negative
sentences = ["i love this movie", "this is a great film", "what a wonderful experience", "i am so happy",
    "this is good stuff", "awesome work really enjoyed it", "best book ever", "a true masterpiece",
    "i hate this film", "this is a bad movie", "what a terrible moment", "i am very sad",
    "this is not good", "awful job hated it", "worst book of my life", "a real disaster"
]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

words = set(word for sentence in sentences for word in sentence.split())
vocab = {word:i+1 for i,word in enumerate(words)}
vocab['<pad>'] = 0
vocab_len = len(vocab)
#将文本转为数字列表，并填充
tokenized_sen = [[vocab[word] for word in sentence.split()] for sentence in sentences]
max_len = max(len(s) for s in tokenized_sen)
pad_sen = np.array([s + [vocab["<pad>"]]*(max_len-len(s)) for s in tokenized_sen])
#转化为张量
input_tensor = torch.LongTensor(pad_sen)
label_tensor = torch.LongTensor(labels)
#创建dataset dataloader
dataset = TensorDataset(input_tensor,label_tensor)
dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
#超参数
D_MODEL = config["d_model"]      # 嵌入维度
NUM_HEADS = config["num_heads"]     # 头数
NUM_LAYERS = config["num_layers"]    # Encoder层数
D_FF = config["d_ff"]         # FFN的隐藏层维度 2/4倍嵌入维度
NUM_CLASSES = config["num_classes"]   # 类别数
EPOCHS = config["epochs"]       # 训练轮次
LR = config["learning_rate"]        # 学习率

#设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"now the device is {device}")

#实例化模型 损失函数 优化器
classifier = transformerBlock(NUM_CLASSES,NUM_LAYERS,D_MODEL,D_FF,NUM_HEADS,vocab_len,dropout=0.1)
classifier.to(device) #将模型所有参数和缓冲移动到GPU
Loss_model = nn.CrossEntropyLoss() #交叉熵 用于分类问题，输入是logits
optimizer = torch.optim.Adam(classifier.parameters(),lr=LR)

#训练循环
loss_history = []
print("start training...")
for epoch in range(EPOCHS):
    classifier.train() # 只是告诉模型此时是训练阶段
    epoch_loss = 0
    for batch_input,batch_label in dataloader:
        #将每个批次数据移动到GPU
        batch_input = batch_input.to(device)
        batch_label = batch_label.to(device)

        src_mask = (batch_input!=vocab["<pad>"]).to(device)
        #前向传播
        output = classifier(batch_input,src_mask)
        loss = Loss_model(output,batch_label)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    loss_history.append(epoch_loss/len(dataloader))
print("finish training")

#绘图loss
plt.figure(figsize=(10,5))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss_history)
plt.title("Traininig loss curve")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

#预测并与标签评估
classifier.eval() #进入评估阶段
with torch.no_grad():  #在关闭梯度下降这一状态下
    all_input = input_tensor.to(device)
    predictions = classifier(all_input)
    predictions_label = torch.argmax(predictions,dim=1).cpu()
    all_labels = label_tensor
    
    total = []

    print("\n---prediction VS label---\n")
    for i in range(len(sentences)):
        '''
        print(f"sentence:{sentences[i]}\n")
        print(f"prediction:{'Positive' if predictions_label[i]==1 else 'Negative'}\n")
        print(f"label:{'Positive' if all_labels[i]==1 else 'Negative'}\n")
        '''
        #训练结果和标签相同为1，反之为0
        if predictions_label[i] == all_labels[i]:
            total.append(1)
        else:
            total.append(0)
    print("训练结果和标签相同为1 反之为0")
    print(total)


    
    
