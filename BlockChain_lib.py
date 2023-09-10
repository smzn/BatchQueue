import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

class BlockChain_lib:
    def __init__(self, lmd, mean, sd, time, capacity = 1000000, block_meantime = 60 * 10):
        self.lmd = lmd #トランザクションの到着率
        self.mean = mean #1トランザクションの大きさの平均(正規分布)
        self.sd = sd #1トランザクションの大きさの標準偏差(正規分布)
        self.capacity = 1000000 #1ブロックは1MByteまで
        self.capacity = capacity
        #self.block_meantime = 60 * 10 #(約10分)
        self.block_meantime = block_meantime
        self.time = time # 上限時間
        self.time_series = [] #経過時間
        self.transaction = [] #トランザクション数の時系列推移
        print('ブロックの許容量(MB): {0}'.format(self.capacity))
        print('ブロックの平均処理時間(秒): {0}'.format(self.block_meantime))

    def getEXP(self, param): #paramは指数分布の平均
        return rd.exponential(param)

    def getNormal(self):
        return np.random.normal(self.mean, self.sd)

    def getSimulation(self): # M/G^b/1待ち行列として考える
        block_volume = 0 #現在のブロックの容量
        block_count = 0 #現在のブロックに入っているトランザクション数
        block_service = 100000000 #ブロックサービスまでの時間
        buffer = 0 #バッファにあるトランザクションの数
        queue = 0 #現在のqueueの長さ(ブロック内+バッファのトランザクション数)

        total_queue = 0 #系内人数
        total_queuelength = 0 #待ち人数

        arrival = self.getEXP(1/self.lmd) #最初のトランザクション到着までの時間
        elapse = 0 #経過時間
        
        while(elapse < self.time):
            if(arrival < block_service): #トランザクションの到着
                #print("Arrival 経過時間:{0}".format(elapse))
                #系内トランザクション数の処理
                total_queue += queue * arrival
                if( queue > 0 ):
                    total_queuelength += ( queue - block_count ) * arrival
                elif ( queue == 0 ):
                    total_queuelength += queue * arrival
                queue += 1 #系内のトランザクション数を増やす(サービス中も含む)

                if(buffer == 0): #bufferが0は、ブロックは未生成
                    cur_volume = self.getNormal() #今回のトランザクションの大きさ
                    block_volume += cur_volume #到着トランザクションをブロックに格納
                    block_count += 1 #ブロックのトランザクション格納数を増やす
                    #print("ブロックに格納 ブロック容量:{0}, ブロック内TS数{1}".format(block_volume, block_count))
                    if(block_volume > self.capacity):#ブロックの規定量より大きくなったら
                        block_service = self.getEXP(self.block_meantime) #ブロックのサービス時間を設定
                        #block_count = 1 #ブロック中のトランザクションは今回の到着のみ
                        #block_volume = cur_volume #ブロック中のトランザクションは今回の到着のみ
                        buffer += 1
                        #print("ブロックを処理開始 サービス時間:{0}".format(block_service))

                elif(buffer > 0):#ブロックは生成済み(処理待ち)
                    block_service -= arrival #ブロックのサービス時間を減らす
                    buffer += 1 #今回の到着はバッファに入る

                elapse += arrival #時間を進める
                arrival = self.getEXP(1/self.lmd) #次のトランザクションの到着までの時間

            elif(arrival >= block_service): #サービスが完了
                #print("Departure 経過時間:{0}".format(elapse))
                #系内トランザクション数の処理
                total_queue += queue * block_service
                if(queue > 0):
                    total_queuelength += ( queue - block_count ) * block_service
                elif ( queue == 0 ):
                    total_queuelength += queue * block_service
                queue -= (block_count - 1) #サービスが終わったブロックに入っていたトランザクション数を引く(-1する:確認)
                arrival -= block_service #次回までの到着時刻を変更
                elapse += block_service #時間を進める
                block_count = 0 #サービス終了で初期化
                block_volume = 0 #サービス終了で初期化
                if(buffer > 0): #新たなブロックを到着済みトランザクションで作成
                    for i in range(buffer): #到着済みトランザクションでのループ
                        cur_volume = self.getNormal()
                        block_volume += cur_volume
                        block_count += 1
                        if(block_volume > self.capacity):#ブロックの規定量より大きくなったら
                            block_service = self.getEXP(self.block_meantime) #ブロックのサービス時間を設定
                            #block_count = 1 #ブロック中のトランザクションは今回の到着のみ (要確認!!!!)
                            #block_volume = cur_volume #ブロック中のトランザクションは今回の到着のみ
                            #print("ブロックを処理開始 サービス時間:{0}".format(block_service))
                            break #ループをbreak
                        else :
                            block_service = 1000000 #ブロックが規定量に満たない場合
                    buffer -= block_count #ブロックに入れたトランザクション数をbufferから引く(値確認)  

            #print("系内TS数:{0}, ブロック内TS容量:{1}, ブロック内TS個数{2}, ブロックサービス時間{3}".format(queue,block_volume,block_count,block_service)) 
            self.time_series.append(elapse)
            self.transaction.append(queue)  
        print("平均系内人数:{0}, 平均待ち人数:{1}".format(total_queue / self.time, total_queuelength / self.time))
        #系内：全てのトランザクション、待ち:バッファのトランザクションのみ
        return total_queue / self.time

    def getGraph(self, filename="blockchain_graph.png"):
        plt.plot(self.time_series, self.transaction)
        plt.title("Transition of Number of Transaction")
        plt.xlabel("Time")
        plt.ylabel("Number of transaction")
        plt.grid(True)
        plt.savefig(filename) # This line saves the graph as a file
        plt.show()
