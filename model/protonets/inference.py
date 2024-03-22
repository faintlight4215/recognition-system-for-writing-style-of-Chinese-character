#!/usr/bin/env python
# coding: utf-8
import sys
import cv2 as cv


# 整理一下内部的label类别，label是索引号，后面需要根据索引号索引出来类别进行输出：
English_and_chinese_dictionary = {
	"HYDaLiShuJ"              : 	"汉仪大隶书简体",
	"HYDunHuangXieJingW"      :		"汉仪敦煌写经体",
	"HYGuLiW"                 :		"汉仪古隶",
	"HYShangWeiHeFengTiW"     :		"汉仪尚巍和风体",
	"HYShouJinShuJ"           : 	"汉仪瘦金书简体",
	"HYShuTongTiJ"            : 	"汉仪舒同体简体",
	"HYYanHuShouShuW"         :		"汉仪彦湖手书体",
	"HYYanKaiW"               :		"汉仪颜真卿楷体",
	"HYZengXiangChanQuTiW"    :		"汉仪曾翔禅趣体",
	"HanyiSentyJournal"       :  	"汉仪新蒂手札体",
	"HanyiSentyZHAO"          :		"汉仪新蒂赵孟頫体",
	"FZSuXSHTWBLSJF"          : 	"苏新诗好太王碑隶书",
	"FZSXSLKJF"               :  	"苏新诗柳楷",
	"STFDBTYTFU"              : 	"书体坊颜真卿楷书",
	"STFHSJKJF"               :   	"书体坊何绍基楷书",
	"STFMiFXSFU"              :  	"书体坊米芾行书",
	"STFWangDXSJF"            : 	"书体坊王铎行书",
	"STFZhaoMFXKJF"           :  	"书体坊赵孟頫行楷",
	"FZZJ-HLYHXSFU"           : 	"方正字迹黄陵野鹤",
	"FZBaDSRXKJW"             :  	"方正八大山人行楷",
	"FZCaoQBLSJW"             : 	"方正曹全碑隶书",
	"FZChuSLKSJW"             : 	"方正褚遂良楷书",
	"FZCuanBZBKSJF"           : 	"方正爨宝子碑楷书",
	"FZDongQCXSJW"            : 	"方正董其昌行书",
	"FZHaoTWBLSJW"            : 	"方正好太王碑隶书",
	"FZHuangTJXSJF"           :   	"方正黄庭坚行书",
	"FZLingFJXKJW"            :  	"方正灵飞经小楷",
	"FZLiQBLSJF"              :  	"方正礼器碑隶书",
	"FZLiuBSLSJF"             :   	"方正刘炳森隶书",
	"FZLiuGQKSJF"             :  	"方正柳公权楷书",
	"FZLiYXSJW"               :   	"方正李邕行书",
	"FZLuXXSJF"               :  	"方正鲁迅行书",
	"FZMaWDBSJF"              : 	"方正马王堆帛书",
	"FZMiFXSJW"               : 	"方正米芾行书",
	"FZOuYXKSJF"              :  	"方正欧阳询楷书",
	"FZOuYZSXSJF"             :  	"方正欧阳中石行书",
	"FZQiGXKJF"               :  	"方正启功行楷",
	"FZShenYMXSJF"            : 	"方正沈尹默行书",
	"FZSHiMMKSJW"             : 	"方正石门铭楷书",
	"FZShiMSLSJW"             :  	"方正石门颂隶书",
	"FZShuTXSJF"              :  	"方正舒同行书",
	"FZSuSXSJF"               :   	"方正苏轼行书",
	"FZTaiSJGJLSJF"           :   	"方正泰山金刚经隶书",
	"FZWangXZXKJW"            : 	"方正王献之小楷",
	"FZWangXZXSJF"            : 	"方正王羲之行书",
	"FZWenZMXKJW"             :		"方正文征明小楷",
	"FZWuYRXSJF"              :  	"方正吴玉如行书",
	"FZXiPSJLSJF"             :  	 "方正熹平石经隶书",
	"FZXiXSLSJW"              :  	"方正西狭颂隶书",
	"FZYangNSXSJW"            :  	"方正杨凝式行书",
	"FZYanZQKSJF"             :  	"方正颜真卿楷书",
	"FZYiBSLSJW"              :  	"方正伊秉绶隶书",
	"FZYiYBLSJW"              :   	"方正乙瑛碑隶书",
	"FZZhangMLBKSJW"          : 	"方正张猛龙碑楷书",
	"FZZhangQBLSJW"           : 	"方正张迁碑隶书",
	"FZZhaoJSJSJF"            :  	 "方正赵佶瘦金书",
	"FZZhaoMFKSJF"            :  	"方正赵孟頫楷书",
	"FZZhaoMFXSJF"            :  	"方正赵孟頫行书",
	"FZZHengWGBKSJW"          :		"方正郑文公碑楷书",
	"FZZhiYKSJW"              :  	 "方正智永楷书"
}

# label_and_English_dictionary={
# 	0						:		"FZXiXSLSJW",
# 	1						:		"FZYanZQKSJF",
# 	2						:		"FZYangNSXSJW",
# 	3						:		"FZZhangQBLSJW",
# 	4						:		"FZZhaoMFKSJF",
# 	5						:		"FZZhaoJSJSJF",
# 	6						:		"FZCuanBZBKSJF",
# 	7						:		"FZDongQCXSJW",
# 	8						:		"FZChuSLKSJW",
# 	9						:		"FZBaDSRXKJW",
# 	10						:		"FZCaoQBLSJW",
# 	11						:		"HYShangWeiHeFengTiW",
# 	12						:		"HYShouJinShuJ",
# 	13						:		"HYShuTongTiJ",
# 	14						:		"FZShenYMXSJF",
# 	15						:		"FZQiGXKJF",
# 	16						:		"FZSHiMMKSJW",
# 	17						:		"FZWenZMXKJW",
# 	18						:		"FZXiPSJLSJF",
# 	19						:		"FZWuYRXSJF",
# 	20						:		"FZHuangTJXSJF",
# 	21						:		"FZHaoTWBLSJW",
# 	22						:		"FZLingFJXKJW",
# 	23						:		"s3",
# 	24						:		"lgq",
# 	25						:		"csl",
# 	26						:		"FZYiYBLSJW",
# 	27						:		"FZYiBSLSJW",
# 	28						:		"FZZhangMLBKSJW",
# 	29						:		"FZShuTXSJF",
# 	30						:		"FZSuoJZCFU",
# 	31						:		"FZShiMSLSJW",
# 	32						:		"FZTaiSJGJLSJF",
# 	33						:		"FZWangXZXKJW",
# 	34						:		"FZWangXZXSJF",
# 	35						:		"FZZhaoMFXSJF",
# 	36						:		"FZZhiYKSJW",
# 	37						:		"FZZHengWGBKSJW",
# 	38						:		"zmf",
# 	39						:		"hysw",
# 	40						:		"zqcwx",
# 	41						:		"FZZJ-HLYHXSFU",
# 	42						:		"HanyiSentyJournal",
# 	43						:		"HanyiSentyZHAO",
# 	44						:		"FZLiuBSLSJF",
# 	45						:		"FZLiuGQKSJF",
# 	46						:		"FZLiQBLSJF",
# 	47						:		"FZSuSXSJF",
# 	48						:		"FZSXSLKJF",
# 	49						:		"FZSuXSHTWBLSJF",
# 	50						:		"HYGuLiW",
# 	51						:		"HYDunHuangXieJingW",
# 	52						:		"HYDaLiShuJ",
# 	53						:		"FZLuXXSJF",
# 	54						:		"FZLiYXSJW",
# 	55						:		"FZMaWDBSJF",
# 	56						:		"FZOuYXKSJF",
# 	57						:		"FZOuYZSXSJF",
# 	58						:		"FZMiFXSJW"
# }

label_and_English_dictionary = {
	0						:		"FZSuSXSJF",
	1						:		"FZSXSLKJF",
	2						:		"FZWangXZXKJW",
	3						:		"FZWangXZXSJF",
	4						:		"FZYangNSXSJW",
	5						:		"HYShangWeiHeFengTiW",
	6						:		"FZZJ-HLYHXSFU",
	7						:		"HanyiSentyJournal",
	8						:		"FZZhaoJSJSJF",
	9						:		"HanyiSentyZHAO",
	10						:		"FZShuTXSJF",
	11						:		"FZLuXXSJF",
	12						:		"FZLiYXSJW",
	13						:		"FZOuYXKSJF",
	14						:		"FZMiFXSJW",
	15						:		"FZDongQCXSJW",
	16						:		"FZHuangTJXSJF",
	17						:		"FZChuSLKSJW",
	18						:		"FZBaDSRXKJW",
	19						:		"FZLiuGQKSJF",
	20						:		"HYYanHuShouShuW",
	21						:		"STFWangDXSJF",
	22						:		"HYShuTongTiJ",
	23						:		"STFDBTYTFU",
	24						:		"HYYanKaiW"
}

# sys.path.append(r'E:\Pycharm-project\calligraphy-ratings\protonet\protonets')
sys.path.append('/home/lxy/原型/protonets')
from protonets_net import Protonets
import os
# os.chdir(r'E:\Pycharm-project\calligraphy-ratings\protonet\scripts')
os.chdir('/home/lxy/原型/scripts')	
sys.path.append('/home/lxy/原型/scripts') #这样才能找到load data
import load_data
import numpy as np
from utils import del_files

def load_model():
    # 根据需求修改类的初始化参数，参数含义见protonets_net.py
	wide = 105
	length = 105
	protonets = Protonets((1,wide,length),5,20,20,3,'../log/',step = 10000,trainval=True) # 不训练，只加载模型
	return protonets

def inference_protonet(protonets):
	# 写入图片
	# del_files('/home/lxy/原型/style-input-picture') # 清除上一张图片
	# cv.imwrite('/home/lxy/原型/style-input-picture/1.png', img)  # 写入待识别书写风格的图片,要命名为1.png
	# 载入数据
	_,_,input = load_data.load_data()
	wide = input.shape[0]
	length = input.shape[1]
	# protonets = Protonets((1,wide,length),5,20,20,3,'../log/',step = 10000,trainval=True) # 不训练，只加载模型
	input = np.resize(input, (105, 105))
	return protonets.inference_model(input)

if __name__ == "__main__":
    # 得出了预测的label,对应到索引和字体
	protonets = load_model()
label = inference_protonet(protonets)
print(label_and_English_dictionary[label])
print(English_and_chinese_dictionary[label_and_English_dictionary[label]])