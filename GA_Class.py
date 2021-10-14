import math
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

class GA_Class:
    def __init__(self, npop, ngen, mutpb, weight_limit, path, data_file):
        self.npop = npop
        self.ngen = ngen
        self.mutpb = mutpb
        self.weight_limit = weight_limit
        self.path = path
        self.parcel = self.getCSV(self.path + data_file).values.tolist()
        #print(self.parcel)
        #乱数の初期化
        SEED = 32767
        random.seed(SEED)
        #変数の準備
        self.pool = [[self.getRandInt(2) for i in range(len(self.parcel))] for j in range(self.npop)]      #染色体プール
        self.ngpool = [[0 for i in range(len(self.parcel))] for j in range(self.npop * 2)] #次世代染色体プール
        self.bestfit_seriese = []#最適遺伝子適合度を入れたリスト
        self.mean_bestfit_seriese = [] #遺伝子全体平均の適合度
        
    def getGA(self):
        #打ち切り世代まで繰り返し
        for generation in range(self.ngen):
            print(generation,"世代")
            self.getMating()   #交叉
            self.getMutation()      #突然変異
            self.getSelectng() #次世代の選択
            elite,bestfit,mean = self.getResult()          #結果出力
            print('遺伝子番号 : {0}, 最適遺伝子適合度 : {1}, 平均遺伝子適合度 : {2}'.format(elite, bestfit, mean))
            self.bestfit_seriese.append(bestfit)
            self.mean_bestfit_seriese.append(mean)
        print(self.bestfit_seriese)
        print(self.mean_bestfit_seriese)
        #グラフ描画
        x_axis = [i for i in range(self.ngen)]
        fig = plt.figure()
        plt.plot(x_axis, self.bestfit_seriese, label='elite')
        plt.plot(x_axis, self.mean_bestfit_seriese, label='mean')
        plt.title('Transition of GA Value')
        plt.xlabel('Generation')
        plt.ylabel('Value of GA')
        plt.grid()
        plt.legend()
        fig.savefig('/content/drive/MyDrive/研究/BCMP/GA/graph/ga_transition.png')
            
    # mating()関数
    def getMating(self):
        """交叉"""
        roulette = [0 for i in range(self.npop)] #ルーレット
        #ルーレットの作成
        totalfitness = 0                        #適応度の合計値
        for i in range(self.npop):
            roulette[i] = self.evalfit(self.pool[i])
            #適応度の合計値を計算
            totalfitness += roulette[i]
        #選択と交叉を繰り返す
        for i in range(self.npop): #ここでは遺伝子数回繰り返している
            while True:   #重複の削除
                mama = self.selectp(roulette,totalfitness)
                papa = self.selectp(roulette,totalfitness)
                if mama != papa:
                    break #重複なし
            #特定の２遺伝子の交叉
            self.crossing(self.pool[mama],self.pool[papa]
                     ,self.ngpool[i * 2],self.ngpool[i * 2 + 1])
     
        return 
    
    # evalfit()関数
    def evalfit(self, g):
        """適応度の計算"""
        value = 0         #評価値
        weight = 0        #重量
        #各遺伝子座を調べて重量と評価値を計算
        for pos in range(len(self.parcel)):
            weight += self.parcel[pos][0] * g[pos]
            value  += self.parcel[pos][1] * g[pos]
        #致死遺伝子の処理
        if weight >= self.weight_limit:
            value = 0
        return value
        
    # selectp()関数
    def selectp(self, roulette,totalfitness):
        """親の選択"""
        acc = 0
        ball = self.getRandInt(totalfitness)
        for i in range(self.npop):
            acc += roulette[i]  #適応度を積算
            if acc > ball:
                break           #対応する遺伝子   
        return i
        
    # crossing()関数
    def crossing(self,m,p,c1,c2):
        """特定の２染色体の交叉"""
        #交叉点の決定
        cp =self.getRandInt(len(self.parcel)) 
        #前半部分のコピー
        for j in range(cp):
            c1[j] = m[j]
            c2[j] = p[j]
        #後半部分のコピー
        for j in range(cp,len(self.parcel)):
            c2[j] = m[j]
            c1[j] = p[j]
        return
    
    # mutation()関数
    def getMutation(self):
        """突然変異"""
        for i in range(self.npop*2):
            for j in range(len(self.parcel)):
                if random.random() < self.mutpb:
                    #反転の突然変異
                    self.ngpool[i][j] = 1 if self.ngpool[i][j] == 0 else 0
        return 
    
    # selectng()関数
    def getSelectng(self):
        """次世代の選択"""
        totalfitness = 0                            #適応度の合計値
        roulette = [0 for i in range(self.npop * 2)] #ルーレット
        acc = 0                                     #適応度の積算値
    
        #選択を繰り返す
        for i in range(self.npop):
            #ルーレットの作成
            totalfitness=0
            for c in range(self.npop * 2):
                roulette[c] = self.evalfit(self.ngpool[c])
                #適応度の合計値を計算
                totalfitness += roulette[c]
            #染色体を一つ選ぶ
            ball = self.getRandInt(totalfitness)
            acc = 0
            for c in range(self.npop*2):
                acc += roulette[c]  #適応度を積算
                if acc > ball:
                    break           #対応する遺伝子
            # 染色体のコピー
            self.pool[i] = copy.deepcopy(self.ngpool[c])
        return 
    
    # printp()関数
    def getResult(self):
        """結果出力"""
        totalfitness = 0  #適応度の合計値
        bestfit = 0       #エリート遺伝子の処理用変数
        #染色体プールの出力
        for i in range(self.npop):
            fitness = self.evalfit(self.pool[i])
            if fitness > bestfit: #エリート解
                bestfit = fitness
                elite = i
            #print(pool[i],"  fitness =",fitness)
            totalfitness += fitness
        #エリート解の適応度と平均適応度を出力
        return (elite,bestfit,totalfitness/self.npop)
    
    def getCSV(self, file):
        return pd.read_csv(file, engine='python', encoding='utf-8')
        
    def getRandInt(self, n):#0~n未満の整数の乱数
        return random.randint(0,n-1)  
        
        
if __name__ == '__main__':
    path = '/content/drive/MyDrive/研究/BCMP/'
    data_file = 'GA/data.csv'
    npop = 30
    ngen = 50
    mutpb = 0.1
    weight_limit = 1000
    ga = GA_Class(npop, ngen, mutpb, weight_limit, path, data_file)
    ga.getGA()