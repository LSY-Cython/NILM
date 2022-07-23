import torch.nn
import time
from arcface import *
from preprocess import ClassDataset
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import copy

def training(model,head,epoch,device,batch_size,init_lr):
    with open("../dataset/data.json", "r") as f:
        data = json.load(f)
    train_set = ClassDataset(imgData=data["train"])
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    model.to(device)
    entropy_loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=init_lr)
    epoch_loss = list()
    for epoch_idx in range(epoch):
        batch_loss = list()
        start_time = time.time()
        for imgs,labels in train_loader:
            imgs = imgs.to(device)
            embeddings = model(imgs)
            thetas = head(embeddings,labels)
            loss = entropy_loss(thetas,labels)
            loss_val = loss.item()
            batch_loss.append(loss_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        epoch_loss.append(np.mean(batch_loss))
        print(f"[epoch {epoch_idx}/{epoch}] loss={epoch_loss[-1]:.5f} time={end_time-start_time:.5f}s")
        if (epoch_idx+1)%10==0:
            torch.save(model.state_dict(), f"weights/epoch{epoch_idx+1}.pt")
    plt.plot(epoch_loss, color="blue", label="loss")
    plt.legend()
    plt.savefig(f"train_loss.png")

def testing(model,weight_path,device):
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    with open("../dataset/data.json","r") as f:
        data = json.load(f)
    test_set = ClassDataset(imgData=data["test"])
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    test_embed = {"embeddings":list()}
    with torch.no_grad():
        start_time = time.time()
        for img,label in test_loader:
            img = img.to(device)
            label = label.item()
            embedding = model(img).cpu().numpy().tolist()[0]
            test_embed["embeddings"].append([embedding,label])
        end_time = time.time()
        print(f"单张图像推理平均耗时: {(end_time-start_time)/len(test_loader):.5f}s")
    with open("embed.json","w") as f:
        json.dump(test_embed,f,indent=4)

def evaluation():
    app_embeddings = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    similarity = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    with open("embed.json","r") as f:
        embeddings = json.load(f)["embeddings"]
    for result in embeddings:
        embedding,label = result[0],result[1]
        app_embeddings[label].append(embedding)
    for i in range(8):
        vectors = app_embeddings[i][1:]  # 选取首个embedding作为特征向量
        for v in vectors:
            cos_sim = list()
            for j in range(8):
                metric = cosine_similarity(np.array(app_embeddings[j][0]),np.array(v))
                cos_sim.append(metric)
            similarity[i].append(cos_sim)
    accuracy = list()
    for i in range(8):
        num = len(similarity[i])
        tp = 0
        for item in similarity[i]:
            if np.array(item).argmax()==i and max(item)>=0.75:
                tp+=1
        accuracy.append(f"{i}: {tp}/{num}={tp/num}")
    print("测试准确率：", accuracy)
    with open("similarity.json", "w") as f:
        json.dump(similarity,f,indent=4)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEncoder(chnum_in=1)
    head = Arcface(embedding_size=128,classnum=6,s=64.,m=0.5)
    # training(
    #     model=model,
    #     head=head,
    #     epoch=200,
    #     device=device,
    #     batch_size=4,
    #     init_lr=1e-3
    # )
    testing(model=model,
            weight_path="weights/epoch200.pt",
            device=device)
    evaluation()

