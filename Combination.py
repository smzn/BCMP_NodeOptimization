import numpy as np
import itertools
import sys

R = int(sys.argv[1])
K_total = int(sys.argv[2])

def getCombiList4(K, Pnum): #並列計算用：Pnumを増やしながら並列計算(2022/1/19)
  #Klist各拠点最大人数 Pnum足し合わせ人数
  Klist = [[j for j in range(K[i]+1)] for i in range(len(K))]
  combKlist = list(itertools.product(*Klist))
  combK = [list(cK) for cK in combKlist if sum(cK) == Pnum ]
  return combK

K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
print(K)
val_list = []
for i in range(0,K_total):
  val = getCombiList4(K,i)
  val_list.append(len(val))
  print('index = {0}, number of combination = {1} '.format(i,len(val)))
print('Sum of combination = {0}'.format(sum(val_list)))
print('Max of combination = {0}'.format(max(val_list)))

#python3 Combination.py 3 1250 > Combination_3_1250.txt
