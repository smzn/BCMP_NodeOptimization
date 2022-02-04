import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import sys
import csv
import math
import datetime
from mpi4py import MPI
import itertools

class BCMP_MVA_Computation:
    
    def __init__(self, N, R, K, mu, type_list, m, K_total, rank, size, comm):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.p = self.getTransitionProbability()
        #print(datetime.datetime.now())
        #print(self.p)
        #print(self.p.shape)
        self.alpha = self.getArrival(self.p)
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.m = m #今回は窓口数1(複数窓口は実装できていない)
        self.K_total = K_total
        self.rank = rank
        self.size = size
        self.comm = comm
        print('rank = {0}'.format(self.rank))
        #print('size = {0}'.format(self.size))
        #self.combi_list = []
        #self.combi_list = self.getCombiList2([], self.K, self.R, 0, self.combi_list) #K, Rのときの組み合わせ 
        
        #並列計算で全ての計算過程を保持しないように変更する(2022/01/23)
        self.km = (np.max(self.K)+1)**self.R #[2,1]なので3進法で考える(3^2=9状態) #[0,0]->0, [0,1]->1, [1,0]->3, [1,1]->4, [2,0]->6, [2,1]->7
        #self.L = np.zeros((self.N, self.R, self.km),dtype = 'float32') #平均形内人数 (self.Lを作成するとR=5で作成できなくなる 2022/02/04)
        #self.T = np.zeros((self.N, self.R, self.km),dtype = 'float32') #平均系内時間
        #self.lmd = np.zeros((self.R, self.km),dtype = 'float32') #各クラスのスループット
        
    def getMVA(self):
        state_list = [] #ひとつ前の階層の状態番号
        l_value_list =[] #ひとつ前の状態に対するLの値
        last_L = []
        for k_index in range(1, self.K_total+1):
            if k_index == self.K_total:
                last_L = l_value_list
            print('rank = {0}, k_index = {1}'.format(self.rank, k_index))
            k_combi_list = self.getCombiList4(self.K, k_index)
            #k_combi_listをsize分だけ分割
            k_combi_list_div = [[] for i in range(self.size)]
            size_index = 0
            for k_combi_list_val in k_combi_list:
                k_combi_list_div[size_index % self.size].append(k_combi_list_val)
                size_index += 1
            #print('k_combi_list_all = {0}'.format(k_combi_list_div))
            #print('rank = {0}, k_combi_list_div = {1}'.format(self.rank, k_combi_list_div[self.rank]))
            #for idx, val in enumerate(self.combi_list):
            #print('k_combi_list_div length = {0}'.format(len(k_combi_list_div[self.rank])))
            
            #並列ループ内での受け渡しに利用
            L = np.zeros((self.N, self.R, len(k_combi_list_div[self.rank])),dtype = 'float32') #平均形内人数 
            T = np.zeros((self.N, self.R, len(k_combi_list_div[self.rank])),dtype = 'float32') #平均系内時間
            lmd = np.zeros((self.R, len(k_combi_list_div[self.rank])),dtype = 'float32') #各クラスのスループット

            for idx, val in enumerate(k_combi_list_div[self.rank]):#自分の担当だけ実施
                #Tの更新
                #k_state = self.getState(val) #kの状態を10進数に変換
                #print('Index : {0}, k = {1}, state = {2}'.format(idx, val, k_state))

                for n in range(self.N): #P336 (8.43)
                    for r in range(self.R):
                        if self.type_list[n] == 3:
                            #self.T[n,r, k_state] = 1 / self.mu[r,n]
                            T[n,r, idx] = 1 / self.mu[r,n]
                        else:
                            r1 = np.zeros(self.R) #K-1rを計算
                            r1[r] = 1 #対象クラスのみ1
                            k1v = val - r1 #ベクトルの引き算
                            #print('n = {0}, r = {1}, k1v = {2}, k = {3}'.format(n,r,k1v, val))

                            if np.min(k1v) < 0: #k-r1で負になる要素がある場合
                                continue

                            sum_l = 0
                            for i in range(self.R):#k-1rを状態に変換
                                if np.min(k1v) >= 0: #全ての状態が0以上のとき(一応チェック)
                                    kn = self.getState(k1v)
                                    #sum_l += self.L[n, i, int(kn)] #knを整数型に変換 (self.Lを利用しない 2022/02/04)
                                    #state_list_idx = self.list_index(state_list[n*self.R:(n+1)*self.R], kn) #前回の情報で更新 state_listは現在のn,rの場合で持ってくる (2022/02/04)
                                    state_list_idx = self.list_index(state_list, kn)
                                    #print(state_list[n*self.R:(n+1)*self.R])
                                    #print('今回の状態番号 : {0}'.format(kn))
                                    #print(state_list)
                                    #print('state_list_idx : {0}'.format(state_list_idx))
                                    #print(l_value_list)
                                    if state_list_idx >=0:
                                        #print(l_value_list[n*self.R:(n+1)*self.R][state_list_idx])
                                        #sum_l += l_value_list[n*self.R:(n+1)*self.R][state_list_idx] 
                                        #print('今回のl_value_list[{1}]の値 : {0}'.format(l_value_list[state_list_idx + n * self.R + i],state_list_idx + n * self.R + i))
                                        sum_l += l_value_list[state_list_idx + n * self.R + r] 
                            if self.m[n] == 1: #P336 (8.43) Type-1,2,4 (m_i=1)
                                #print('n = {0}, r = {1}, k_state = {2}'.format(n,r,k_state))
                                #self.T[n, r, k_state] = 1 / self.mu[r, n] * (1 + sum_l)
                                T[n, r, idx] = 1 / self.mu[r, n] * (1 + sum_l)
                #print('T = {0}'.format(self.T))

                #λの更新
                for r in range(self.R):
                    sum = 0
                    for n in range(self.N):
                        #sum += self.alpha[r,n] * self.T[n,r,k_state]
                        sum += self.alpha[r,n] * T[n,r,idx]
                    if sum > 0:
                        #self.lmd[r,k_state] = val[r] / sum
                        lmd[r,idx] = val[r] / sum
                    #print('r = {0}, k = {1},lambda = {2}'.format(r, val, self.lmd[r,k_state]))

                #Gの更新
                ''' #rの扱いをどうしたらいい？(要確認)
                r1 = np.zeros(R) #K-1rを計算
                r1[r] = 1 #対象クラスのみ1
                k1v = val - r1 #ベクトルの引き算
                kn = getState(K,R,k1v)
                print('kn = {0}'.format(kn))
                print('lamda = {0}'.format())
                G[k_state] = G[int(kn)] / lmd[r,int(kn)]
                '''

                #Lの更新
                for n in range(self.N):#P336 (8.47)
                    for r in range(self.R):
                        #self.L[n,r,k_state] = self.lmd[r,k_state] * self.T[n,r,k_state] * self.alpha[r,n]
                        L[n,r,idx] = lmd[r,idx] * T[n,r,idx] * self.alpha[r,n]
                        #print('n = {0}, r = {1}, k = {2}, L = {3}'.format(n,r,val,L[n,r,k_state]))

                ''' self.Lは利用しない (2022/02/04)
                #aggregation to self.T,lmd,L at self.rank == 0
                if self.rank == 0:
                    for idx, j in enumerate(k_combi_list_div[self.rank]):
                        k_state = self.getState(j) 
                        #for n in range(self.N): #Update T (Tとλはまとめる必要がない)
                        #    for r in range(self.R):
                        #        self.T[n, r, k_state] = T[n, r, idx]
                        #for r in range(self.R): #Update Lambda
                        #    self.lmd[r,k_state] = lmd[r,idx]
                        for n in range(self.N):#Update L
                            for r in range(self.R):
                                self.L[n,r,k_state] = L[n,r,idx]
                '''        
            #全体の処理を集約してからブロードキャスト
            state_list = []
            l_value_list =[]
            if self.rank == 0:
                for idx, j in enumerate(k_combi_list_div[0]): #rank == 0の情報をまとめる
                    k_state = self.getState(j)
                    for n in range(self.N):#Lの更新
                        for r in range(self.R):
                            state_list.append(k_state) #self.Lの代わりにこれを渡す(2022/02/03)
                            l_value_list.append(L[n,r,idx]) #self.Lの代わりにこれを渡す(2022/02/03)
                for i in range(1, self.size):
                    #k_combi_list_div_rank = self.comm.recv(source=i, tag=11)
                    #T_rank = self.comm.recv(source=i, tag=12)
                    #lmd_rank = self.comm.recv(source=i, tag=13)
                    l_rank = self.comm.recv(source=i, tag=14) #Lのみ集約
                    #print('receive : {0}, {1}'.format(i, l_rank))
                    
                    #リストの結合
                    for idx, j in enumerate(k_combi_list_div[i]):
                        #k_state = self.getState(k_combi_list_div_rank[j]) #kの状態を10進数に変換
                        k_state = self.getState(j) #kの状態を10進数に変換
                        #for n in range(self.N): #Tの更新
                        #    for r in range(self.R):
                        #        self.T[n, r, k_state] = T_rank[n, r, idx]
                        #for r in range(self.R): #Lambdaの更新
                        #    self.lmd[r,k_state] = lmd_rank[r,idx]
                        for n in range(self.N):#Lの更新
                            for r in range(self.R):
                                #self.L[n,r,k_state] = l_rank[n,r,idx] #利用しない(2022/02/04)
                                state_list.append(k_state) #self.Lの代わりにこれを渡す(2022/02/03)
                                l_value_list.append(l_rank[n,r,idx]) #self.Lの代わりにこれを渡す(2022/02/03)
                #self.comm.barrier() #プロセス同期
                #print(self.T)
                #print(self.lmd)
                #print(self.L)
            
            else:
                #self.comm.send(k_combi_list_div[self.rank], dest=0, tag=11)
                #self.comm.send(T, dest=0, tag=12)
                #self.comm.send(lmd, dest=0, tag=13)
                self.comm.send(L, dest=0, tag=14)
            self.comm.barrier() #プロセス同期
            
            #ここでブロードキャストする
            #self.T = self.comm.bcast(self.T, root=0)
            #self.lmd = self.comm.bcast(self.lmd, root=0)
            #self.L = self.comm.bcast(self.L, root=0) #self.Lをブロードキャストするとエラーになるのでやめる(2022/02/03)
            state_list = self.comm.bcast(state_list, root=0)
            l_value_list = self.comm.bcast(l_value_list, root=0)
            ''' self.Lを利用しない 2022/02/04
            if self.rank != 0: #各プロセスでself.Lに集約(2022/02/03)
                for n in range(self.N):
                    for r in range(self.R):
                        for i,j in zip(state_list, l_value_list): #このfor文はいらない？単に一つずつindexを増やして入れればいい
                            self.L[n,r,i] = j
             '''
            
        #平均系内人数最終結果
        #last = self.getState(self.combi_list[-1]) #combi_listの最終値が最終結果の人数
        #last = self.getState(self.K) #combi_listの最終値が最終結果の人数 -> 利用しない(2022/02/04)
        #L_index = {'class0': self.L[:,0,last], 'class1' : self.L[:,1,last]} #クラス2個の場合
        #L_index = {'class0': self.L[:,0,last]} #クラス1つの場合
        #df_L = pd.DataFrame(L_index)
        #df_L.to_csv('/content/drive/MyDrive/研究/BCMP/csv/MVA_L(N:'+str(self.N)+',R:'+str(self.R)+',K:'+str(self.K)+').csv')
        #return self.L[:,:,last]
        return last_L
            
    def list_index(self, l, val, default=-1):#リスト内に値がある場合、その要素番号を返す、ない場合は-1を返す(2022/02/04)
        if val in l:
            return l.index(val)
        else:
            return default    
        
    def getState(self, k):#k=[k1,k2,...]を引数としたときにn進数を返す(R = len(K))
        k_state = 0
        for i in range(self.R): #Lを求めるときの、kの状態を求める(この例では3進数)
            k_state += k[i]*((np.max(self.K)+1)**(self.R-1-i))
        return k_state

    def getArrival(self, p):#マルチクラスの推移確率からクラス毎の到着率を計算する
        p = np.array(p) #リストからnumpy配列に変換(やりやすいので)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R): #マルチクラス毎取り出して到着率を求める
            alpha[r] = self.getCloseTraffic(p[r * self.N : (r + 1) * self.N, r * self.N : (r + 1) * self.N])
        return alpha
    
    #閉鎖型ネットワークのノード到着率αを求める
    #https://mizunolab.sist.ac.jp/2019/09/bcmp-network1.html
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) #次元を1つ小さくする
        pe = p[1:len(p), 1:len(p)].T - e #行と列を指定して次元を小さくする
        lmd = p[0, 1:len(p)] #0行1列からの値を右辺に用いる
        try:
            slv = solve(pe, lmd * (-1)) #2021/09/28 ここで逆行列がないとエラーが出る
        except np.linalg.LinAlgError as err: #2021/09/29 Singular Matrixが出た時は、対角成分に小さい値を足すことで対応 https://www.kuroshum.com/entry/2018/12/28/python%E3%81%A7singular_matrix%E3%81%8C%E8%B5%B7%E3%81%8D%E3%82%8B%E7%90%86%E7%94%B1
            print('Singular Matrix')
            pe += e * 0.00001 
            slv = solve(pe, lmd * (-1)) 
        #lmd *= -1
        #slv = np.linalg.pinv(pe) * lmd #疑似逆行列で求める
        alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return alpha    

    def getCombiList2(self, combi, K, R, idx, Klist):
        if len(combi) == R:
            #print(combi)
            Klist.append(combi.copy())
            #print(Klist)
            return Klist
        for v in range(K[idx]+1):
            combi.append(v)
            Klist = self.getCombiList2(combi, K, R, idx+1, Klist)
            combi.pop()
        return Klist
    
    def getCombiList4(self, K, Pnum): #並列計算用：Pnumを増やしながら並列計算(2022/1/19)
        #Klist各拠点最大人数 Pnum足し合わせ人数
        Klist = [[j for j in range(K[i]+1)] for i in range(len(K))]
        combKlist = list(itertools.product(*Klist))
        combK = [list(cK) for cK in combKlist if sum(cK) == Pnum ]
        return combK

    def getTransitionProbability(self):
        pr = np.zeros((self.R*self.N, self.R*self.N))
        for r in range(self.R):
            class_number = 0
            while class_number != 1:
                p = np.random.rand(self.N, self.N)
                for i, val in enumerate(np.sum(p, axis=1)):
                    p[i] /= val
                for i in range(self.N):
                    for j in range(self.N):
                        pr[r*self.N+i,r*self.N+j] = p[i,j]
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
    


if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3]) 
    #N = 33 #33
    #R = 2
    #K_total = 500
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    m = np.full(N, 1) #今回は窓口数1(複数窓口は実装できていない)
    #p = pd.read_csv('/content/drive/MyDrive/研究/BCMP/csv/transition33.csv')
    #bcmp = BCMP_MVA(N, R, K, mu, type_list, p, m)
    bcmp = BCMP_MVA_Computation(N, R, K, mu, type_list, m, K_total, rank, size, comm)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    if rank == 0:
        print('L = \n{0}'.format(L))
        for n in range(N):
            for r in range(R):
                print('L[{0},{1},{3}] = {2}'.format(n, r, L[(n*R+r)],n*R+r))
    
    #mpiexec -n 8 python3 BCMP_MVA_Computation.py 33 2 100
