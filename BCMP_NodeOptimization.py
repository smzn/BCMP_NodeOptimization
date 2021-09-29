import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
import sys

class BCMP_NodeOptimization:
    
    def __init__(self, N, R, K_total, path, popularity_file, distance_file, node_number):
        self.N = N
        self.R = R
        self.K_total = K_total
        self.K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
        self.mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
        self.type_list = np.full(N, 1) #サービスタイプはFCFS
        self.m = np.full(N, 1) #今回は窓口数1(複数窓口は実装できていない)
        self.path = path
        sys.path.append(path)
        #拠点人気度(クラス別)、拠点間距離の取り込み
        popularity = self.getCSV(self.path + popularity_file)
        #self.node = list(popularity['Id']) #利用するノード
        self.popularity = popularity.iloc[:,2:4].values.tolist() #人気度をリストに変換
        self.distance = self.getCSV(self.path + distance_file)
        self.node_number = node_number
        
    def getCSV(self, file):
        return pd.read_csv(file, engine='python', encoding='utf-8')
        
    #距離行列作成関数
    def getDistance(self, node):
        distance_matrix = np.zeros((len(node),len(node)))
        for row in self.distance.itertuples(): #右三角行列で作成される
            if row.fromid in node and row.toid in node:
                distance_matrix[node.index(int(row.fromid))][node.index(int(row.toid))] = row.distance
        for i in range(len(distance_matrix)): #下三角に値を入れる(対称)
            for j in range(i+1, len(distance_matrix)):
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix
        
    #重力モデルで推移確率行列を作成 
    def getGravity(self, distance): #distanceは距離行列(getDistanceで作成)、popularityはクラス分の人気度
      C = 0.1475
      alpha = 1.0
      beta = 1.0
      eta = 0.5
      class_number = len(self.popularity[0]) #クラス数
      tp = np.zeros((len(distance) * class_number, len(distance) * class_number))
      for r in range(class_number):
        for i in range(len(distance) * r, len(distance) * (r+1)):
          for j in range(len(distance) * r, len(distance) * (r+1)):
            if distance[i % len(distance)][j % len(distance)] > 0:
              tp[i][j] = C * (self.popularity[i % len(distance)][r]**alpha) * (self.popularity[j % len(distance)][r]**beta) / (distance[i % len(distance)][j % len(distance)]**eta)
      row_sum = np.sum(tp, axis=1) #行和を算出
      for i in range(len(tp)): #行和を1にする
        if row_sum[i] > 0:
          tp[i] /= row_sum[i]
      return tp
      
    def getObjective(self, l, capacity, pnenalty):
      sum = 0
      cnt = 0
      for i in range(len(l)):
        sum += np.abs(l[i] - capacity[i])
        if l[i] > capacity[i]:
          sum += pnenalty[i]
          cnt += 1
      return sum
      
    def getOptimizeGA(self, individual):
      #1. 利用ノードリストの作成
      node = []
      for i, val in enumerate(individual):
        if val == 1:
          node.append(i)
      #2. 利用ノードでの距離行列作成
      distance_matrix = self.getDistance(node)
      #3. 利用ノードでの推移確率行列の作成(重力モデルの利用)
      p = self.getGravity(distance_matrix)
      #逆行列を持つか確認
      #equivalence, class_number = self.getEquivalence(0, 100, p)#0は閾値、5はステップ数
      #if class_number > 1:
      #    return 1000000
      #4. 定常分布を求める
      import BCMP_MVA as mdl
      bcmp_mva = mdl.BCMP_MVA(len(node), self.R, self.K, self.mu, self.type_list, p, self.m)
      theoretical = bcmp_mva.getMVA()
      #5. 目的関数の評価
      capacity = np.full(self.N, self.K_total // self.N + 1) #各拠点のキャパ：今は平均(ノード数の変更なし)
      pnenalty = np.full(self.N , self.K_total * 0.05) #ペナルティ値は網内人数の5%としてみる
      L_class = np.array(theoretical) #numpy形式に変換
      L = [] #クラスを拠点にまとめる
      for i in range(len(L_class)):
        sum = 0
        for j in range(len(L_class[i])):
          sum += L_class[i,j]
        L.append(sum)
      return self.getObjective(L, capacity, pnenalty),      

    #遺伝子の制約をつける
    #https://deap.readthedocs.io/en/master/tutorials/advanced/constraints.html?highlight=constraint
    def feasible(self, individual):
        sum = 0
        for i in individual:
          sum += i
        if self.node_number <= sum:
            return True
        return False
      
    def getGA(self, npop, ngen): #個体数、世代数
        creator.create( "Fitness", base.Fitness, weights=(-1.0,) )#-1.0の場合は最小化
        creator.create("Individual", list, fitness = creator.Fitness ) #重複がない場合はsetを使う
        toolbox = base.Toolbox()
        toolbox.register( "attribute", random.randint, 0, 1 ) #遺伝子は0,1で構成
        toolbox.register( "individual", tools.initRepeat, creator.Individual, toolbox.attribute, self.N ) #初期個体の生成(遺伝子の長さはN)
        toolbox.register( "population", tools.initRepeat, list, toolbox.individual ) #初期個体群を作成          
        toolbox.register("evaluate", self.getOptimizeGA)
        toolbox.decorate("evaluate", tools.DeltaPenalty(self.feasible, 100000.0))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register( "mate", tools.cxTwoPoint )
        toolbox.register( "mutate", tools.mutFlipBit, indpb=0.05 )
        
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        pop = toolbox.population(n=npop)#個体数
        algorithms.eaSimple( pop, toolbox, cxpb = 0.8, mutpb=0.5, ngen=ngen, stats=stats, halloffame=hof, verbose=True)#世代数指定
        best_ind = tools.selBest(pop, 1)[0]
        #print("最も良い個体は %sで、そのときの目的関数の値は %s" % (best_ind, best_ind.fitness.values))
        return best_ind, best_ind.fitness.values
        
    #同値類を求める関数
    def getEquivalence(self, th, roop, p):
        list_number = 0 #空のリストを最初から使用する

        #1. 空のリストを作成して、ノードを追加しておく(作成するのはノード数分)
        equivalence = [[] for i in range(len(p))] 
        
        #2. Comunicationか判定して、Commnicateの場合リストに登録
        for ix in range(roop):
            p = np.linalg.matrix_power(p.copy(), ix+1) #累乗
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    if(p[i][j] > th and p[j][i] > th): #Communicateの場合
                        #3. Communicateの場合登録するリストを選択
                        find = 0 #既存リストにあるか
                        for k in range(len(p)):
                            if i in equivalence[k]: #既存のk番目リストに発見(iで検索)
                                find = 1 #既存リストにあった
                                if j not in equivalence[k]: #jがリストにない場合登録
                                    equivalence[k].append(j)        
                                break
                            if j in equivalence[k]: #既存のk番目リストに発見(jで検索)
                                find = 1 #既存リストにあった
                                if i not in equivalence[k]:
                                    equivalence[k].append(i)        
                                break
                        if(find == 0):#既存リストにない
                            equivalence[list_number].append(i)
                            if(i != j):
                                equivalence[list_number].append(j)
                            list_number += 1

        #4. Communicateにならないノードを登録
        for i in range(len(p)):
            find = 0
            for j in range(len(p)):
                if i in equivalence[j]:
                    find = 1
                    break
            if find == 0:
                equivalence[list_number].append(i)
                list_number += 1

        #5. エルゴード性の確認(class数が1のとき)
        class_number = 0
        for i in range(len(p)):
            if len(equivalence[i]) > 0:
                class_number += 1

        return equivalence, class_number
    
    
        
if __name__ == '__main__':
    
    path = '/content/drive/MyDrive/研究/BCMP/'
    N = int(sys.argv[1]) #全体拠点数
    R = int(sys.argv[2]) #クラス数
    K_total = int(sys.argv[3]) #網内人数
    node_number = int(sys.argv[4]) #拠点利用数
    npop = int(sys.argv[5]) #遺伝子数
    ngen = int(sys.argv[6]) #世代数
    popularity_file = 'TransitionProbability/csv/popularity2.csv'
    distance_file = 'TransitionProbability/csv/distance.csv'
    bcmp = BCMP_NodeOptimization(N, R, K_total, path, popularity_file, distance_file, node_number)
    opt_node, obj = bcmp.getGA(npop, ngen)
    print('Optimized Node : {0}'.format(opt_node))
    print('Optimized ObjectiveFunction : {0}'.format(obj))
    
    