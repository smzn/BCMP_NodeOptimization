import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import sys
import csv
import math

class BCMP_MVA:
    
    def __init__(self, N, R, K, mu, type_list, p, m):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.p = p
        self.alpha = self.getArrival(self.p)
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.m = m #今回は窓口数1(複数窓口は実装できていない)
        self.combi_list = []
        self.combi_list = self.getCombiList2([], self.K, self.R, 0, self.combi_list) #K, Rのときの組み合わせ 
        self.km = (np.max(self.K)+1)**self.R #[2,1]なので3進法で考える(3^2=9状態) #[0,0]->0, [0,1]->1, [1,0]->3, [1,1]->4, [2,0]->6, [2,1]->7
        self.L = np.zeros((self.N, self.R, self.km)) #平均形内人数 
        self.T = np.zeros((self.N, self.R, self.km)) #平均系内時間
        self.lmd = np.zeros((self.R, self.km)) #各クラスのスループット
        
        
    def getMVA(self):
        for idx, val in enumerate(self.combi_list):
            if idx == 0:
                continue
            #Tの更新
            k_state = self.getState(val) #kの状態を10進数に変換
            #print('Index : {0}, k = {1}, state = {2}'.format(idx, val, k_state))
            
            for n in range(self.N): #P336 (8.43)
                for r in range(self.R):
                    if self.type_list[n] == 3:
                        self.T[n,r, k_state] = 1 / self.mu[r,n]
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
                                sum_l += self.L[n, i, int(kn)] #knを整数型に変換
                        if self.m[n] == 1: #P336 (8.43) Type-1,2,4 (m_i=1)
                            #print('n = {0}, r = {1}, k_state = {2}'.format(n,r,k_state))
                            self.T[n, r, k_state] = 1 / self.mu[r, n] * (1 + sum_l)
            #print('T = {0}'.format(T))

            #λの更新
            for r in range(self.R):
                sum = 0
                for n in range(self.N):
                    sum += self.alpha[r,n] * self.T[n,r,k_state]
                if sum > 0:
                    self.lmd[r,k_state] = val[r] / sum
                #print('r = {0}, k = {1},lambda = {2}'.format(r, val, lmd[r,k_state]))
            
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
                    self.L[n,r,k_state] = self.lmd[r,k_state] * self.T[n,r,k_state] * self.alpha[r,n]
                    #print('n = {0}, r = {1}, k = {2}, L = {3}'.format(n,r,val,L[n,r,k_state]))
        
        #平均系内人数最終結果
        last = self.getState(self.combi_list[-1]) #combi_listの最終値が最終結果の人数
        #L_index = {'class0': self.L[:,0,last], 'class1' : self.L[:,1,last]} #クラス2個の場合
        L_index = {'class0': self.L[:,0,last]} #クラス1つの場合
        df_L = pd.DataFrame(L_index)
        #df_L.to_csv('/content/drive/MyDrive/研究/BCMP/csv/MVA_L(N:'+str(self.N)+',R:'+str(self.R)+',K:'+str(self.K)+').csv')
        return self.L[:,:,last]
            
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

if __name__ == '__main__':
    
    #推移確率行列に合わせる
    N = 33 #33
    R = 2
    K_total = 500
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    m = np.full(N, 1) #今回は窓口数1(複数窓口は実装できていない)
    p = pd.read_csv('/content/drive/MyDrive/研究/BCMP/csv/transition33.csv')
    bcmp = BCMP_MVA(N, R, K, mu, type_list, p, m)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    print('L = \n{0}'.format(L))