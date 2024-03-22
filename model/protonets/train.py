#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/lxy/原型/protonets')	
from protonets_net import Protonets

import os 
os.chdir('/home/lxy/原型/scripts')	
sys.path.append('/home/lxy/原型/scripts') #这样才能找到load data
import load_data

import csv
import numpy as np
import random
import torch

if __name__ == '__main__':
	##载入数据
	labels_trainData ,labels_testData,_ = load_data.load_data()
	class_number = max(list(labels_trainData.keys()))+1 # 这个地方应该有加一吧，应该是9个class
	
	print(class_number)
	wide = labels_trainData[0][0].shape[0]
	length = labels_trainData[0][0].shape[1]
	
	print(wide,length)
	for label in labels_trainData.keys():  # 统一形状
		labels_trainData[label] = np.reshape(labels_trainData[label], [-1,1,wide, length])
	for label in labels_testData.keys():
		labels_testData[label] = np.reshape(labels_testData[label], [-1,1,wide, length])
	
	##初始化模型
	# 加载训练好的模型
	protonets = Protonets((1,wide,length),15,20,20,3,'../log/',10000)	#'''根据需求修改类的初始化参数，参数含义见protonets_net.py'''

	# train
	#训练prototypical_network
	for n in range(9001):	 ##随机选取x个类进行一个episode的训练，一种训练1000个epoch
		# 看看提升epoch能不能train的更好一点
		protonets.train(labels_trainData,class_number)
		if n % 100 == 0 and n != 0:	#每50次存储一次模型，并测试模型的准确率，训练集的准确率和测试集的准确率被存储在model_step_eval.txt中
			torch.save(protonets.model, '../log/model_net_'+str(n)+'.pkl')
			protonets.save_center('../log/model_center_'+str(n)+'.csv')
			train_accury,test_accury,a,c= protonets.evaluation_model(labels_testData,labels_trainData)
			# print(train_accury,test_accury,a,c)
			print(f"n = {n}        train_accury: {train_accury}        test_accury: {test_accury}")
			str_data = str(n) + ',' + str('     train_accury     ') + str(train_accury) + str('       test_accury     ') + str(test_accury) + '\n'
			with open('../log/model_step_eval.txt', "a") as f:
				f.write(str_data)
		# print(n)

		if n == 9000:
			train_accury,test_accury,predict1,j2label_c = protonets.evaluation_model(labels_testData,labels_trainData)
			for key in j2label_c.items():
				n=key[1]+1
				m=abs(round(predict1.tolist()[key[0]],3))
				n1=str(n)
				m1=str(m)
				a="C:\\Users\\24292\\Desktop\\zhinengfenbie\\fonts\\"+n1+".TTF"
				b="C:\\Users\\24292\\Desktop\\zhinengfenbie\\fonts\\"+n1+"："+ m1+".TTF"
				# a="/home/lxy/原型/scripts/zhinengfenbie/fonts/"+n1+".TTF"
				# b="/home/lxy/原型/scripts/zhinengfenbie/fonts/"+n1+"："+ m1+".TTF"
				os.rename(a,b)  

