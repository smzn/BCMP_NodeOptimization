import math
#import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from numpy.random import randint
from numpy.random import rand
from mpi4py import MPI

#ノードをランダムに与え、計算を実施する(2022/10/03)

class BCMP_GA_Class:
    def __init__(self, N, R, K_total, node_number, npop, ngen, crosspb, mutpb, rank, size, comm):
        self.N = N
        self.R = R
        self.K_total = K_total
        self.K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
        self.mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
        self.type_list = np.full(N, 1) #サービスタイプはFCFS
        self.m = np.full(N, 1) #今回は窓口数1(複数窓口は実装できていない)
        #self.path = path
        #sys.path.append(path)
        #popularity = self.getCSV(popularity_file) #拠点人気度(クラス別)
        #self.popularity = popularity.iloc[:,2:4].values.tolist() #人気度をリストに変換
        #self.distance = self.getCSV(distance_file) #拠点間距離の取り込み
        self.node_number = node_number #最低利用拠点数
        self.npop = npop
        self.ngen = ngen
        self.crosspb = crosspb
        self.mutpb = mutpb #突然変異率
        self.rank = rank
        self.size = size
        self.comm = comm
        self.scores = [0 for i in range(self.npop)] #各遺伝子のスコア
        self.bestfit_seriese = []#最適遺伝子適合度を入れたリスト
        self.mean_bestfit_seriese = [] #遺伝子全体平均の適合度
        #print('rank = {0}, size = {1}'.format(self.rank, self.size))
        #初期遺伝子をブロードキャスト
        if self.rank == 0:
            self.pool = [[self.getRandInt1() for i in range(self.N)] for j in range(self.npop)] #遺伝子を初期化
        else:
            self.pool = [[]]
        self.pool = self.comm.bcast(self.pool, root=0)
        #print(self.pool)
        
    #https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # genetic algorithm
    def genetic_algorithm(self): #objective, n_bits, n_iter, n_pop, r_cross, r_mut
        #if self.rank == 0:
        #    best, best_eval = 0, self.getOptimizeBCMP(self.pool[0])
        best, best_eval = self.pool[0], 10**5
        for gen in range(self.ngen):
            self.scores = [0 for i in range(self.npop)] #2021/11/01 世代ずつ初期化(各rankで計算しない遺伝子が集約時に足されてしまう)
            if self.rank == 0:
                print('{0}世代'.format(gen))
            #ここからrank担当遺伝子のみ計算をしていく
            #遺伝子の最小利用拠点数のチェック
            for pop_index in range(self.rank, self.npop, self.size):#自分の担当遺伝子だけループ 
                #print('rank = {0}, npop = {1}'.format(self.rank, pop_index))
                if sum(self.pool[pop_index]) < self.node_number: #最低利用拠点を下回ったら初期化
                    self.pool[pop_index] = [self.getRandInt1() for i in range(self.N)]
                start = time.time()
                self.scores[pop_index] = self.getOptimizeBCMP(self.pool[pop_index])
                elapsed_time = time.time() - start
                print ("rank = {1}, pop_id = {2}, calclation_time:{0}".format(elapsed_time, self.rank, pop_index) + "[sec]")
                print('pop_index = {0}, scores = {1}, rank = {2}'.format(pop_index, self.scores[pop_index], self.rank))
            
            '''#全体でやる場合(並列化無し)
            for i in range(self.npop):
                if sum(self.pool[i]) < self.node_number: #最低利用拠点を下回ったら初期化
                    self.pool[i] = [self.getRandInt1() for i in range(self.N)]
                    #print('初期化')
            self.scores = [self.getOptimizeBCMP(c) for c in self.pool]
            '''
            #データの集約 (ここまでチェックする 10/20)
            if self.rank == 0:
                for i in range(1, self.size):
                    scores = self.comm.recv(source=i, tag=11)
                    #print('receive : {0}, {1}'.format(i, scores))
                    #リストの結合
                    for j in range(len(self.scores)):
                        self.scores[j] += scores[j]
            else:
                self.comm.send(self.scores, dest=0, tag=11)
            self.comm.barrier() #プロセス同期
            if self.rank == 0:
            #    print('Aggregation scores')
                print('{1}世代評価 : {0}'.format(self.scores, gen))
            #並列化ここまで(遺伝子情報を集約する)
            
            
            #評価はrank0が行ってブロードキャストする
            # check for new best solution
            if self.rank == 0:
                for i in range(self.npop):
                    if self.scores[i] < best_eval and sum(self.pool[i]) >= self.node_number: #最小値を探す
                        best, best_eval = self.pool[i], self.scores[i]
                        print("{0}世代, new best {1} = {2}".format(gen, self.pool[i], self.scores[i]))
                        print('拠点利用数 : {0}'.format(sum(self.pool[i])))
                # select parents
                selected = [self.selection() for c in range(self.npop)] 
                # create the next generation
                children = list()
                for i in range(0, self.npop, 2):
                    # get selected parents in pairs
                    p1, p2 = selected[i], selected[i+1] #遺伝子数が奇数だとエラー
                    # crossover and mutation
                    for c in self.crossover(p1, p2):
                        # mutation
                        self.mutation(c)
                        # store for next generation
                        children.append(c)
                # replace population
                self.pool = children
                #世代毎の目的関数値を保存
                self.bestfit_seriese.append(best_eval)
                self.mean_bestfit_seriese.append(sum(self.scores)/len(self.scores))
            #ここでブロードキャストする
            self.pool = self.comm.bcast(self.pool, root=0)
            best = self.comm.bcast(best, root=0)
            best_eval = self.comm.bcast(best_eval, root=0)
            #print('遺伝子同期(rank = {0}) : {1}'.format(self.rank, self.pool))
            
        if self.rank == 0:
            self.getGraph()
            #20211230 Final Result
            self.getFinalResult(best)
        return [best, best_eval]
    	  
    def getFinalResult(self, individual):
        #1.Node List
        node = []
        for i, val in enumerate(individual):
            if val == 1:
                node.append(i)
        #2. Distance Matrix
        #distance_matrix = self.getDistance(node)
        #3. Transition Matrix with a gravity model
        #p = self.getGravity(distance_matrix)
        p = self.getTransitionProbability(len(node))
        #4. Stationary Distribution
        import BCMP_MVA as mdl
        bcmp_mva = mdl.BCMP_MVA(len(node), self.R, self.K, self.mu, self.type_list, p, self.m)
        theoretical = bcmp_mva.getMVA()
        #5. Objective Function
        capacity = np.full(self.N, self.K_total // self.N + 1) 
        pnenalty = np.full(self.N , self.K_total * 0.05) 
        L_class = np.array(theoretical) #list to numpy
        np.savetxt('./ga_L_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.node_number)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', L_class, delimiter=',')
        np.savetxt('./ga_Node_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.node_number)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', individual, delimiter=',')
        np.savetxt('./ga_P_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.node_number)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', p, delimiter=',')
        print('Final Result')
        print('L = {0}'.format(L_class))
        print('Node = {0}'.format(individual))
        print('Node Number = {0}'.format(node))
        print('P = {0}'.format(p))
    	    
    def getOptimizeBCMP(self, individual):
        #1. 利用ノードリストの作成
        node = []
        for i, val in enumerate(individual):
            if val == 1:
                node.append(i)
        #2. 利用ノードでの距離行列作成
        #distance_matrix = self.getDistance(node)
        #3. 利用ノードでの推移確率行列の作成(重力モデルの利用)
        #p = self.getGravity(distance_matrix)
        #推移確率は自動生成にする(2022/10/03)
        p = self.getTransitionProbability(len(node))
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
        return self.getObjective(L, capacity, pnenalty)   

    #nodeが可変なのでローカル変数とする(2022/10/03)
    def getTransitionProbability(self, node):
        pr = np.zeros((self.R*node, self.R*node))
        for r in range(self.R):
            class_number = 0
            while class_number != 1:
                p = np.random.rand(node, node)
                for i, val in enumerate(np.sum(p, axis=1)):
                    p[i] /= val
                for i in range(node):
                    for j in range(node):
                        pr[r*node+i,r*node+j] = p[i,j]
                equivalence, class_number = self.getEquivalence(0, 5, p)
                if class_number == 1:
                    break
        return pr

    def getEquivalence(self, th, roop, p):
        list_number = 0 

        #1.
        equivalence = [[] for i in range(len(p))] 
        
        #2.
        for ix in range(roop):
            p = np.linalg.matrix_power(p.copy(), ix+1) 
            for i in range(len(p)):
                for j in range(i+1, len(p)):
                    if(p[i][j] > th and p[j][i] > th):
                        #3. 
                        find = 0 
                        for k in range(len(p)):
                            if i in equivalence[k]:
                                find = 1 
                                if j not in equivalence[k]:
                                    equivalence[k].append(j)        
                                break
                            if j in equivalence[k]: 
                                find = 1 
                                if i not in equivalence[k]:
                                    equivalence[k].append(i)        
                                break
                        if(find == 0):
                            equivalence[list_number].append(i)
                            if(i != j):
                                equivalence[list_number].append(j)
                            list_number += 1

        #4.
        for i in range(len(p)):
            find = 0
            for j in range(len(p)):
                if i in equivalence[j]:
                    find = 1
                    break
            if find == 0:
                equivalence[list_number].append(i)
                list_number += 1

        #5.
        class_number = 0
        for i in range(len(p)):
            if len(equivalence[i]) > 0:
                class_number += 1

        return equivalence, class_number


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
    
    # tournament selection
    def selection(self, k=3):
    	# first random selection
    	selection_ix = randint(self.npop)
    	for ix in randint(0, self.npop, k-1):
    		# check if better (e.g. perform a tournament)
    		if self.scores[ix] < self.scores[selection_ix]:
    			selection_ix = ix
    	return self.pool[selection_ix]
     
    # crossover two parents to create two children
    def crossover(self, p1, p2):
    	# children are copies of parents by default
    	c1, c2 = p1.copy(), p2.copy()
    	# check for recombination
    	if rand() < self.crosspb:
    		# select crossover point that is not on the end of the string
    		pt = randint(1, len(p1)-2)
    		# perform crossover
    		c1 = p1[:pt] + p2[pt:]
    		c2 = p2[:pt] + p1[pt:]
    	return [c1, c2]
     
    # mutation operator
    def mutation(self, bitstring):
    	for i in range(len(bitstring)):
    		# check for a mutation
    		if rand() < self.mutpb:
    			# flip the bit
    			bitstring[i] = 1 - bitstring[i]
 
    def getGraph(self):
        #グラフ描画
        x_axis = [i for i in range(self.ngen)]
        fig = plt.figure()
        plt.plot(x_axis, self.bestfit_seriese, label='elite')
        plt.plot(x_axis, self.mean_bestfit_seriese, label='mean')
        #plt.plot(x_axis, self.max_pool_value_seriese, label='max')
        plt.title('Transition of GA Value')
        plt.xlabel('Generation')
        plt.ylabel('Value of GA')
        plt.grid()
        plt.legend()
        fig.savefig('./ga_transition_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.node_number)+'_'+str(self.npop)+'_'+str(self.ngen)+'.png')
        
        
         

    def getCSV(self, file):
        return pd.read_csv(file, engine='python', encoding='utf-8')
        
    def getRandInt1(self): #1を返すときに最低利用拠点数での1の返しやすさを反映
        if self.node_number / self.N > np.random.rand():
            return 1
        else:
            return 0
        
        
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #path = '/content/drive/MyDrive/研究/BCMP/'
    N = int(sys.argv[1]) #全体拠点数
    R = int(sys.argv[2]) #クラス数
    K_total = int(sys.argv[3]) #網内人数
    node_number = int(sys.argv[4]) #拠点利用数
    npop = int(sys.argv[5]) #遺伝子数
    ngen = int(sys.argv[6]) #世代数
	#推移確率は自動的に与える
    #popularity_file = './csv/popularity2.csv'
    #distance_file = './csv/distance.csv'
    crosspb = 0.5
    mutpb = 0.2
    start = time.time()
    bcmp = BCMP_GA_Class(N, R, K_total, node_number, npop, ngen, crosspb, mutpb, rank, size, comm)
    best, score = bcmp.genetic_algorithm()
    if rank == 0:
        print('Done!')
        print('f(%s) = %f' % (best, score))
        elapsed_time = time.time() - start
        print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
#実行方法
#mpiexec -n 6 python3 BCMP_GA_Class.py 33(N) 2(R) 100(K) 20(最低利用拠点数) 18(遺伝子数) 10(世代数)
    
