#!/usr/bin/env python
# coding: utf-8
import os 
import matplotlib.image as mpimg
import numpy as np
import csv
#将图片数据转化为numpy，每一个类得数据被为训练集和测试集，并存储在字典中

#每个类别一个中心点，感觉类别的数目似乎是有错误的，现在是9个class但是8个中心点
os.chdir('/home/lxy/原型/scripts')
def load_data():
	labels_trainData = {}
	label = 0
	# print("train dataset:")
	for file in os.listdir('/home/lxy/原型/styletrain/'):
		for dir in os.listdir('/home/lxy/原型/styletrain/' + file):
			labels_trainData[label] = []
			data = []
			for png in os.listdir('/home/lxy/原型/styletrain/' + file +'/' + dir):
				if png=="gen_list.json":
					continue  # 不处理json，直接跳过，只处理图像
				image_np = mpimg.imread('/home/lxy/原型/styletrain/' + file +'/' + dir+'/' +png)
				# image_np.resize(105,105)
				image_np = np.resize(image_np, (105, 105)) #因为这个地方debug时候报错，所以改成这样
				image_np.astype(np.float64)
				data.append(image_np)
			labels_trainData[label] = np.array(data)
			# print(f"{label}:{dir}")
			label += 1
	# test dataset的顺序和train dataset label的顺序是一样的，但是读出来就是要和对应的label匹配
	labels_testData = {}
	label = 0
	# print("test_dataset:")
	for file in os.listdir('/home/lxy/原型/styletest/'):
		for dir in os.listdir('/home/lxy/原型/styletest/' + file):
			labels_testData[label] = []
			data = []
			for png in os.listdir('/home/lxy/原型/styletest/' + file +'/' + dir):
				if png=="gen_list.json":
					continue  # 不处理json，直接跳过，只处理图像
				image_np = mpimg.imread('/home/lxy/原型/styletest/' + file +'/' + dir+'/' +png)
				# image_np.resize(105,105)
				image_np = np.resize(image_np, (105, 105))
				image_np.astype(np.float64)
				data.append(image_np)
			labels_testData[label] = np.array(data)
			# print(f"{label}:{dir}")
			label += 1           

	# 读入inference input
	image_np = mpimg.imread('/home/lxy/原型/style-input-picture/1.png')  # 固定的地址和命名格式
	# image_np.resize(105,105)
	image_np = np.resize(image_np, (105, 105))
	image_np.astype(np.float64)

	return labels_trainData ,labels_testData,image_np
