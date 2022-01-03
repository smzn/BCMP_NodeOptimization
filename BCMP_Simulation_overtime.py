import numpy as np
import pandas as pd
import math
import random

class BCMP_Simulation:
    
    def __init__(self, N, R, K, mu, type_list, p, time, path, capacity, capacityclass):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.mu = mu #サービス率 FCFSはクラス別で変えられない
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.p = p
        #print(self.p.shape)
        self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
        self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
        self.eventqueue = [[] for i in range(N)] #各拠点でイベント発生時のqueueの長さ
        self.eventtime = [[] for i in range(N)] #各拠点でイベントが発生した時の時刻
        self.timerate = np.zeros((self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る
        #self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.timerateclass = np.zeros((self.R, self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.time = time #シミュレーション時間
        self.path = path
        self.overtime = np.zeros((self.N, sum(self.K)+1)) #拠点でキャパを超えた時の人数に対する累積時間(3000時刻以上で計測)
        self.overtimeclass = np.zeros((self.R, self.N, sum(self.K)+1))#拠点でキャパを超えた時の人数に対する累積時間(クラス別)(3000時刻以上で計測)
        self.overtime_start = 3000 
        self.capacity = capacity
        self.capacityclass = capacityclass
        #print(self.p.iloc[0,1])
        
        
    def getSimulation(self):
        queue = np.zeros(self.N) #各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
        queueclass = np.zeros((self.N, self.R)) #各拠点のサービス中を含むqueueの長さ(クラス別)
        classorder = [[] for i in range(self.N)] #拠点に並んでいる順のクラス番号
        service = np.zeros(self.N) #サービス中の客の残りサービス時間
        total_length = np.zeros(self.N) #各拠点の延べ系内人数(クラスをまとめたもの)
        total_lengthclass = np.zeros((self.N, self.R)) #各拠点の延べ人数(クラス別)
        total_waiting = np.zeros(self.N) #延べ待ち人数(クラスをまとめたもの)
        total_waitingclass = np.zeros((self.N, self.R))#延べ待ち人数(クラス別)
        L = np.zeros(self.N) #平均系内人数(結果)
        Lc = np.zeros((self.N, self.R)) #平均系内人数(結果)(クラス別)
        Q = np.zeros(self.N) #平均待ち人数(結果)
        Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果)(クラス別)
        rmse = [] #100単位時間でのrmseの値を格納
        rmse_time = [] #rmseを登録した時間
        regist_time = 50 #rmseの登録時刻
        regist_span = 50 #50単位で登録
        
        elapse = 0
        initial_node = 0
        #Step1 開始時の客の分配 (開始時のノードは拠点番号0)
        for i in range(R):
            for j in range(K[i]):
                self.event[initial_node].append("arrival")
                self.eventclass[initial_node].append(i) #到着客のクラス番号
                self.eventqueue[initial_node].append(queue[initial_node])#イベントが発生した時のqueueの長さ(到着客は含まない)
                self.eventtime[initial_node].append(elapse) #(移動時間0)
                queue[initial_node] +=1 #最初はノード0にn人いるとする
                queueclass[initial_node][i] += 1 #拠点0にクラス別人数を追加
                classorder[initial_node].append(i)#拠点0にクラス番号を追加
        service[initial_node] = self.getExponential(self.mu[initial_node]) #先頭客のサービス時間設定
       
        '''
        print('Step1 開始時の客の分配 (開始時のノードは拠点番号0)')
        print('event : {0}'.format(self.event))
        print('eventclass : {0}'.format(self.eventclass))
        print('eventqueue : {0}'.format(self.eventqueue))
        print('eventtime : {0}'.format(self.eventtime))
        print('queue : {0}'.format(queue))
        print('queueclass : {0}'.format(queueclass))
        print('classorder : {0}'.format(classorder))
        print('service : {0}'.format(service))
        '''
        
        #print('Simulation Start')
        #Step2 シミュレーション開始
        while elapse < self.time:
            #print('経過時間 : {0} / {1}'.format(elapse, self.time))
            mini_service = 100000#最小のサービス時間
            mini_index = -1 #最小のサービス時間をもつノード
           
            #print('Step2.1 次に退去が起こる拠点を検索')
            #Step2.1 次に退去が起こる拠点を検索
            for i in range(self.N):#待ち人数がいる中で最小のサービス時間を持つノードを算出
                if queue[i] > 0:
                    if mini_service > service[i]:
                        mini_service = service[i]
                        mini_index = i
            departure_class = classorder[mini_index].pop(0) #退去する客のクラスを取り出す(先頭の客)
            
            '''
            print('現在時刻(elapse) : {0}'.format(elapse))
            print('最小のサービス時間(mini_service) : {0}'.format(mini_service))
            print('最小のサービス時間を持つ拠点番号(mini_index) : {0}'.format(mini_index))
            print('最小のサービス時間を持つ拠点のクラス(departure_class) : {0}'.format(departure_class))
            '''
            
            #Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)
            for i in range(self.N):#ノードiから退去(全拠点で更新)
                total_length[i] += queue[i] * mini_service #ノードでの延べ系内人数
                for r in range(R): #クラス別延べ人数更新
                    total_lengthclass[i,r] += queueclass[i,r] * mini_service
                if queue[i] > 0: #系内人数がいる場合(サービス中の客がいるとき)
                    service[i] -= mini_service #サービス時間を減らす
                    total_waiting[i] += ( queue[i] - 1 ) * mini_service #ノードでの延べ待ち人数
                    for r in range(R):
                        if queueclass[i,r] > 0: #クラス別延べ待ち人数の更新
                            total_waitingclass[i,r] += ( queueclass[i,r] - 1 ) * mini_service 
                elif queue[i] == 0: #いらないかも
                    total_waiting[i] += queue[i] * mini_service
                self.timerate[i, int(queue[i])] += mini_service #人数分布の時間帯を更新
                for r in range(R):
                    #self.timerateclass[i, r, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
                    self.timerateclass[r, i, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
                    
                #キャパオーバーの登録
                if elapse >= self.overtime_start:
                    if self.capacity[i] < queue[i]:
                        self.overtime[i, int(queue[i])] += mini_service #人数分布の時間帯を更新
                    for r in range(R):
                        if self.capacityclass[r, i] < queueclass[i,r]:
                            self.overtimeclass[r, i, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
                    
            '''
            print('Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)')
            print('queue : {0}'.format(queue))
            print('queueclass : {0}'.format(queueclass))
            #print('total_length : {0}'.format(total_length))
            #print('total_lengthclass : {0}'.format(total_lengthclass))
            #print('timerate : {0}'.format(self.timerate))
            #print('timerateclass : {0}'.format(self.timerateclass))
            '''
        
            #Step2.3 退去を反映
            self.event[mini_index].append("departure") #退去を登録
            self.eventclass[mini_index].append(departure_class)
            self.eventqueue[mini_index].append(queue[mini_index]) #イベント時の系内人数を登録
            #self.eventqueueclass[mini_index, departure_class].append(queueclass[mini_index, departure_class]) #イベント時の系内人数を登録
            queue[mini_index] -= 1 #ノードの系内人数を減らす
            queueclass[mini_index, departure_class] -= 1 #ノードの系内人数を減らす(クラス別)
            elapse += mini_service
            self.eventtime[mini_index].append(elapse) #経過時間の登録はイベント後
            if queue[mini_index] > 0:
                service[mini_index] = self.getExponential(self.mu[mini_index])#退去後まだ待ち人数がある場合、サービス時間設定
   
            
            #Step2.4 退去客の行き先決定
            #推移確率行列が N*R × N*Rになっている。departure_class = 0の時は最初のN×N (0~N-1の要素)を見ればいい
            #departure_class = 1の時は (N~2N-1の要素)、departure_class = 2の時は (2N~3N-1の要素)
            #departure_class = rの時は (N*r~N*(r+1)-1)を見ればいい
            rand = random.random()
            sum_rand = 0
            destination_index = -1
            pr = np.zeros((self.N, self.N))#今回退去する客クラスの推移確率行列を抜き出す
            for i in range(self.N * departure_class, self.N * (departure_class + 1)):
                for j in range(self.N * departure_class, self.N * (departure_class + 1)):
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p.iloc[i,j]
            '''
            print(pr)
            print(pr.shape)
            '''
            
            for i in range(len(pr)):    
                sum_rand += pr[mini_index][i]
                if rand < sum_rand:
                    destination_index = i
                    break
            if destination_index == -1: #これは確率が1になってしまったとき用
                destination_index = len(pr) -1 #一番最後のノードに移動することにする
            self.event[destination_index].append("arrival") #イベント登録
            self.eventclass[destination_index].append(departure_class) #移動する客クラス番号登録
            self.eventqueue[destination_index].append(queue[destination_index])
            self.eventtime[destination_index].append(elapse) #(移動時間0)
            #推移先で待っている客がいなければサービス時間設定(即時サービス)
            if queue[destination_index] == 0:
                service[destination_index] = self.getExponential(self.mu[destination_index])
            queue[destination_index] += 1 #推移先の待ち行列に並ぶ
            queueclass[destination_index][departure_class] += 1 #推移先の待ち行列(クラス別)に登録 
            classorder[destination_index].append(departure_class)
            
            '''
            print('Step2.4 退去客の行き先決定')
            print('destination_index : {0}'.format(destination_index))
            print('queue : {0}'.format(queue))
            print('queueclass : {0}'.format(queueclass))
            print('classorder : {0}'.format(classorder))
            '''
           
            #Step2.5 RMSEの計算
            if elapse > regist_time:
                rmse_sum = 0
                theoretical_value = theoretical.values
                lc = total_lengthclass / elapse #今までの時刻での平均系内人数
                for n in range(self.N):
                    for r in range(self.R):
                        rmse_sum += (theoretical_value[n,r] - lc[n,r])**2
                rmse_sum /= self.N * self.R
                rmse_value = math.sqrt(rmse_sum)
                rmse.append(rmse_value)
                rmse_time.append(regist_time)
                regist_time += regist_span
                print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value))
                if rmse_value < sum(self.K) * 0.01: #全人数の1%を下回ったら終了
                    self.time = elapse #途中で終了するのでシミュレーション時間を変更
                    break
            
        L = total_length / self.time
        Lc = total_lengthclass / self.time
        Q = total_waiting / self.time
        Qc = total_waitingclass / self.time
        
        print('平均系内人数L : {0}'.format(L))
        print('平均系内人数(クラス別)Lc : {0}'.format(Lc))
        print('平均待ち人数Q : {0}'.format(Q))
        print('平均待ち人数(クラス別)Qc : {0}'.format(Qc))
       
        pd.DataFrame(L).to_csv(self.path +'/csv/L(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        pd.DataFrame(Lc).to_csv(self.path +'/csv/Lc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        pd.DataFrame(Q).to_csv(self.path +'/csv/Q(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        pd.DataFrame(Qc).to_csv(self.path +'/csv/Qc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        rmse_index = {'time': rmse_time, 'RMSE': rmse}
        df_rmse = pd.DataFrame(rmse_index)
        df_rmse.to_csv(self.path +'/csv/RMSE(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        pd.DataFrame(self.timerate).to_csv(self.path +'/csv/TimeRate(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        for r in range(self.R):
            pd.DataFrame(self.timerateclass[r]).to_csv(self.path +'/csv/TimeRateClass('+str(r)+')(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        pd.DataFrame(self.overtime).to_csv(self.path +'/csv/OverTime(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        for r in range(self.R):
            pd.DataFrame(self.overtimeclass[r]).to_csv(self.path +'/csv/OvertimeClass('+str(r)+')(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+').csv')
        
        
    def getExponential(self, param):
        return - math.log(1 - random.random()) / param 
    
    
    
if __name__ == '__main__':
    
    path = '/sqfs/work/G15281/v60550/BCMP'
    #推移確率行列に合わせる
    N = 33 #33
    R = 2
    K_total = 500
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full(N, 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    p = pd.read_csv(path +'/csv/transition33.csv')
    theoretical = pd.read_csv(path +'/csv/Theoretical33_K500.csv')
    time = 50005
    capacity = np.full(N, K_total // N + 1) #各拠点のキャパ：今は平均
    capacityclass = np.full((R, N), K_total / 2 // N + 1) #各拠点のキャパ(クラス別)：今は平均
    bcmp = BCMP_Simulation(N, R, K, mu, type_list, p, time, path, capacity, capacityclass) 
    bcmp.getSimulation()
