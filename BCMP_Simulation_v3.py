import numpy as np
import pandas as pd
import math
import random
import sys
import time
from mpi4py import MPI

class BCMP_Simulation:
    
    def __init__(self, N, R, K, K_total, mu, type_list, sim_time, rank, size, comm):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.K_total = K_total
        self.mu = mu #サービス率 FCFSはクラス別で変えられない
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.rank = rank
        self.size = size
        self.comm = comm
        self.time = sim_time #シミュレーション時間
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(sum(self.K))+'_Time'+str(self.time)+'_Core'+str(self.size)+'.txt'
        #推移確率生成と共有
        p = [[]]
        if rank == 0:
            p = self.getTransitionProbability(self.N, self.R)
        self.p = comm.bcast(p, root=0)
        with open(self.process_text, 'a') as f:
            print('推移確率行列の同期: rank = {0}, p = {1}'.format(self.rank, self.p), file=f)
        self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
        self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
        self.eventqueue = [[] for i in range(N)] #各拠点でイベント発生時のqueueの長さ
        self.eventtime = [[] for i in range(N)] #各拠点でイベントが発生した時の時刻
        self.timerate = np.zeros((self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る
        self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.start = time.time()
        
        
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
                initial_node = random.randrange(self.N)#20220320 最初はランダムにいる拠点を決定
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
                    self.timerateclass[i, r, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
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
                    #pr[i - self.N * departure_class, j - self.N * departure_class] = self.p.iloc[i,j]
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p[i,j]
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
            regist_time += regist_span
           
        
        #Step2.5 RMSEの計算
        self.comm.barrier() #プロセス同期
        with open(self.process_text, 'a') as f:
            print('RMSEの処理(elapse={1}), rank = {0}'.format(self.rank,elapse), file=f)
        mean_Lc = np.zeros((self.N, self.R))
        Lc_rank = np.zeros((self.size-1, self.N, self.R))
        rmse_sum = 0
        #各プロセスの平均系内人数を集約
        if self.rank == 0:
            #theoretical_value = self.theoretical.values
            mean_Lc = total_lengthclass / elapse #今までの時刻での平均系内人数
            with open(self.process_text, 'a') as f:
                print('MeanLc = {0}, rank = {1}, elapse = {2}'.format(mean_Lc, self.rank, elapse), file=f)
            for i in range(1, self.size):#他のプロセスの情報
                Lc_rank[i-1] = self.comm.recv(source=i, tag=14)
                #mean_Lc += self.comm.recv(source=i, tag=14) #Lのみ集約
                mean_Lc += Lc_rank[i-1] #Lのみ集約
                with open(self.process_text, 'a') as f:
                    print('MeanLc = {0}, rank = {1}, elapse = {2}'.format(mean_Lc, i, elapse), file=f)
        else:
            with open(self.process_text, 'a') as f:
                print('Send Message, rank = {0}, elapse = {1}, Lc = {2}'.format(self.rank, elapse, total_lengthclass / elapse),file=f)
            self.comm.send(total_lengthclass / elapse, dest=0, tag=14)
        if self.rank == 0:
            mean_Lc = mean_Lc / self.size #現在のクラス別平均系内人数
            with open(self.process_text, 'a') as f:
                print('集約したmean_Lc = {0}, rank = {1}, elapse = {2}'.format(mean_Lc, self.rank, elapse),file=f)
            for i in range(1, self.size):
                for n in range(self.N):
                    for r in range(self.R):
                        with open(self.process_text, 'a') as f:
                            print('To receive Lc from {0}'.format(i),file=f)
                        #lc_rank = self.comm.recv(source=i, tag=14)
                        with open(self.process_text, 'a') as f:
                            print('Receive Lc from {0}'.format(i),file=f)
                        #rmse_sum += (mean_Lc[n,r] - lc_rank[n,r])**2
                        rmse_sum += (mean_Lc[n,r] - Lc_rank[i-1,n,r])**2
            rmse_sum /= self.N * self.R
            rmse_value = math.sqrt(rmse_sum)
            rmse.append(rmse_value)
            rmse_time.append(regist_time)
            print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
            print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value))
            print('Elapse = {0}, Lc = {1}'.format(elapse, mean_Lc))
            with open(self.process_text, 'a') as f:
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value), file=f)
                print('Elapse = {0}, Lc = {1}'.format(elapse, mean_Lc), file=f)
                
        L = total_length / self.time
        Lc = total_lengthclass / self.time
        Q = total_waiting / self.time
        Qc = total_waitingclass / self.time
        
        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
        print('平均系内人数L : {0}'.format(L))
        print('平均系内人数(クラス別)Lc : {0}'.format(Lc))
        print('平均待ち人数Q : {0}'.format(Q))
        print('平均待ち人数(クラス別)Qc : {0}'.format(Qc))
        if self.rank == 0:
            with open(self.process_text, 'a') as f:
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                print('平均系内人数L : {0}'.format(L), file=f)
                print('平均系内人数(クラス別)Lc : {0}'.format(Lc), file=f)
                print('平均待ち人数Q : {0}'.format(Q), file=f)
                print('平均待ち人数(クラス別)Qc : {0}'.format(Qc), file=f)
                   
        pd.DataFrame(L).to_csv('./csv/L(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Lc).to_csv('./csv/Lc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Q).to_csv('./csv/Q(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Qc).to_csv('./csv/Qc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        rmse_index = {'time': rmse_time, 'RMSE': rmse}
        df_rmse = pd.DataFrame(rmse_index)
        df_rmse.to_csv('./csv/RMSE(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        
    def getExponential(self, param):
        return - math.log(1 - random.random()) / param 
    
    def getGravity(self, distance, popularity): #distanceは距離行列、popularityはクラス分の人気度
        C = 0.1475
        alpha = 1.0
        beta = 1.0
        eta = 0.5
        class_number = len(popularity[0]) #クラス数
        tp = np.zeros((len(distance) * class_number, len(distance) * class_number))
        for r in range(class_number):
            for i in range(len(distance) * r, len(distance) * (r+1)):
                for j in range(len(distance) * r, len(distance) * (r+1)):
                    if distance[i % len(distance)][j % len(distance)] > 0:
                        tp[i][j] = C * (popularity[i % len(distance)][r]**alpha) * (popularity[j % len(distance)][r]**beta) / (distance[i % len(distance)][j % len(distance)]**eta)
        row_sum = np.sum(tp, axis=1) #行和を算出
        for i in range(len(tp)): #行和を1にする
            if row_sum[i] > 0:
                tp[i] /= row_sum[i]
        return tp
    
    def getTransitionProbability(self, N, R): #20220313追加
        #(1) 拠点の設置と拠点間距離
        node_position_x = [random.randint(0,500) for i in range(N)]
        node_position_y = [random.randint(0,500) for i in range(N)]
        from_id = [] #DF作成用
        to_id = [] #DF作成用
        distance = []
        for i in range(N):
            for j in range(i+1,N):
                from_id.append(i)
                to_id.append(j)
                distance.append(np.sqrt((node_position_x[i]-node_position_x[j])**2 + (node_position_y[i]-node_position_y[j])**2))
        df_distance = pd.DataFrame({ 'from_id' : from_id, 'to_id' : to_id, 'distance' : distance })#データフレーム化
        #距離行列の作成
        distance_matrix = np.zeros((N,N))
        for row in df_distance.itertuples(): #右三角行列で作成される
            distance_matrix[int(row.from_id)][int(row.to_id)] = row.distance
        for i in range(len(distance_matrix)): #下三角に値を入れる(対象)
            for j in range(i+1, len(distance_matrix)):
                distance_matrix[j][i] = distance_matrix[i][j]

        #(2)人気度の設定
        popularity = np.abs(np.random.normal(10, 2, (N, R)))
        
        #(3)推移確率行列の作成
        tp = self.getGravity(distance_matrix, popularity)
        
        #(4)拠点情報(拠点番号、位置(x,y)、人気度(クラス数分))の生成
        df_node = pd.DataFrame({ 'node_number' : range(N), 'position_x' : node_position_x, 'position_y' : node_position_y})
        df_node.set_index('node_number',inplace=True)
        #popularityを追加
        columns = ['popurarity_'+str(i) for i in range(R)]
        for i, val in enumerate(columns):
            df_node[val] = popularity[:, i]
            
        #(5)情報の保存
        df_node.to_csv('./tp/node_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        df_distance.to_csv('./tp/distance_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(distance_matrix).to_csv('./tp/distance_matrix_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(tp).to_csv('./tp/transition_probability_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_Core'+str(self.size)+'.csv', index=True)
        
        return tp
    
    
if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    sim_time = int(sys.argv[4])
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full(N, 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    bcmp = BCMP_Simulation(N, R, K, K_total, mu, type_list, sim_time, rank, size, comm) 
    start = time.time()
    bcmp.getSimulation()
    elapsed_time = time.time() - start
    print ("rank : {1}, calclation_time:{0}".format(elapsed_time, rank) + "[sec]")
    #並列計算用
    #mpiexec -n 4 python3 BCMP_Simulation_v3.py 33 1 500 100 > result_33_1_500_100.txt &