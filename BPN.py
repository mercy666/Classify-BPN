#
from __future__ import division
import math
import random
import string
import pickle
import numpy as np
import pandas as pd

flowerLables = {0:'Iris-setosa',
                1:'Iris-versicolor',
                2:'Iris-virginica'}
random.seed(0)
# 生成隨機數之函數用在初始weight
def rand(a, b):
    return (b-a)*random.random() + a

# 生成x*y之大小之矩陣用在儲存各層weight值
def makeMatrix(x, y, fill=0.0):
    z = []
    for i in range(x):
        z.append([fill]*y)
    return z

# Sigmoid function
def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

class NN: #三層類神經網路
    
    def __init__(self, ni, nh, no): #定義Input 、Hidden、Output node
        self.ni = ni + 1 # 增加一個誤差節點
        self.nh = nh
        self.no = no
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 建立每層之間權重()
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        for i in range(self.ni): #Input與Hidden之間權重random介於(-0.2,0.2)
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2) 
        for j in range(self.nh):#Hidden與Output之間權重random介於(-2.0,2.0)
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        # 建立每層之間Momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('Input layer error')

        #開始Input Layer
        for i in range(self.ni-1):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]

        # 開始Hidden Layer
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 開始Output Layer
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

    
        return self.ao[:]

    def backPropagate(self, targets, N, M):

        # 計算Output Layer 誤差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 計算Hidden Layer 誤差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新 Output Layer weight
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j] #
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # 更新 Input Layer weight
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # 算出誤差
        error = 0.0
        # for k in range(len(targets)):
        #     error = error + 0.5*(targets[k]-self.ao[k])**2
        error += 0.5*(targets[k]-self.ao[k])**2
        return error
    
    def train(self, patterns, iterations=500, N=0.1, M=0.01):#
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('Error %.3f%%' % error)

    def test(self, patterns):
        count = 0
        for p in patterns: 
            target = flowerLables[(p[1].index(1))] #測試資料之真正類別
            result = self.update(p[0])
            index = result.index(max(result)) #將測試資料最後判斷為哪類寫給index      
            print(p[0], ':', target, '->', flowerLables[index])  #將測試結果印出
            count += (target == flowerLables[index]) #若判斷正確則count+1
        accuracy = float(count/len(patterns)) 
        print('accuracy: %.3f' % accuracy) #印出正確率

    def weights(self):
        print('Input weight:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weight:')
        for j in range(self.nh):
            print(self.wo[j])

   
def main():
    data = []
    # 讀檔
    raw = pd.read_csv('C:\Users\kirk\Desktop\iris_new.csv')
    raw_data = raw.values
    raw_feature = raw_data[0:,0:4]
    for i in range(len(raw_feature)):
        xAndY = []
        xAndY.append(list(raw_feature[i]))
        if raw_data[i][4] == 'Iris-setosa':
           xAndY.append([1,0,0]) #Iris-setosa類的標註[1,0,0]
        elif raw_data[i][4] == 'Iris-versicolor':
            xAndY.append([0,1,0]) #Iris-versicolor類的標註[0,1,0]
        else:
            xAndY.append([0,0,1])  
        data.append(xAndY)        #Iris-vergicina類的標註[0,0,1]
       
    #Data先隨機排列
    random.shuffle(data)
    nn = NN(4,7,3)#Input node 4個，hidden node 7個，Output node 3個
    nn.train(data[0:150],iterations=500)
    # 紀錄最後權重
    with open('C:\Users\kirk\Desktop\wi.txt', 'w') as wif:
        pickle.dump(nn.wi, wif)
    with open('C:\Users\kirk\Desktop\wo.txt', 'w') as wof:
        pickle.dump(nn.wo, wof)
    
    nn.test(data[151:])

if __name__ == '__main__':
    main()
