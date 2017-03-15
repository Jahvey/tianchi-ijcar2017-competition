'''
	desc: feature engineering
	author: zhpmatrix@datarush
	date: 2017-01-18
	mod: 2017-02-10@zhpmatrix
	add:
		1.standard deviation of the last two weeks
		2.max value of the last month
'''

from preProcess import *
import numpy as np
import pandas as pd

def _min(sequence):
	'''
		desc: find min
		param: 
			sequence: list
	'''
	if len(sequence) < 1:
		return None
	else:
		return min(sequence)
def _max(sequence):
	'''
		desc: find max
		param: 
			sequence: list
	'''
	if len(sequence) < 1:
		return None
	else:
		return max(sequence)
def avg(sequence):
	'''
		desc: compute average
		param:
			sequence: list
	'''
	if len(sequence) < 1:
		return None
	else:
		return sum(sequence) / len(sequence)  
def median(sequence):
	'''
		desc: find the median value of the list
		param:
			sequence: list
	'''
	if len(sequence) < 1:
		return None
	else:
		sequence.sort()
		return sequence[len(sequence) // 2]

def m2M(data,x):
	'''
		desc: compute the middle value of the last two months for x
		param:
			data: sorted by time_stamp
			x: datetime
	'''
	pays = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	monthDays = 30
	for i in range(1,monthDays*2):
		if loc-i < 0:
			break
		pays.append(data.loc[loc-i]['pay_sum'])
	return median(pays)
def w2A(data,x):
	'''
		desc: compute average value of the last two weeks
		param:
			data: sorted by time_stamp
			x: datetime
	'''
	pays = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	weekDays = 7
	for i in range(1,weekDays*2):
		if loc-i < 0:
			break
		pays.append(data.loc[loc-i]['pay_sum'])
	return avg(pays)

def w2Std(data,x):
	'''
		desc: compute standard deviation value of the last two weeks
		param:
			data: sorted by time_stamp
			x: datetime
	'''
	pays = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	weekDays = 7
	for i in range(1,weekDays*2):
		if loc-i < 0:
			break
		pays.append(data.loc[loc-i]['pay_sum'])
	return np.std(pays)
def m1L(data,x):
	'''
		desc: compute the least valus of the last month
		param: 
			x: datetime
	'''
	pays = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	monthDays = 30
	for i in range(1,monthDays*1):
		if loc-i < 0:
			break
		pays.append(data.loc[loc-i]['pay_sum'])
	return _min(pays)

def m1Lar(data,x):
	'''
		desc: compute the largest value of the last month
		param: 
			x: datetime
	'''
	pays = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	monthDays = 30
	for i in range(1,monthDays*1):
		if loc-i < 0:
			break
		pays.append(data.loc[loc-i]['pay_sum'])
	return _max(pays)
def d3w3(data,x):
	'''
		desc: compute the average value of the last 3 days and divided by the last 3 weeks
		param:
			x: datetime
	'''
	pays3D = []
	pays3W = []
	loc = list(data['time_stamp']).index(pd.Timestamp(x))
	weekDays = 7
	for i in range(1,weekDays*3):
		if loc-i < 0:
			break
		if i<= 3:
			pays3D.append(data.loc[loc-i]['pay_sum'])
			pays3W.append(data.loc[loc-i]['pay_sum'])
		else:
			pays3W.append(data.loc[loc-i]['pay_sum'])
	return float(avg(pays3D))/avg(pays3W)

def getHoliday(data,x):
	loc = data['time_stamp'].tolist().index(pd.Timestamp(x).strftime("%Y-%m-%d"))
	return data.iloc[loc]['holiday']

def getWeather(weather,city,date):
	han2pin = PinYin()
    	han2pin.load_word()
	cityName = han2pin.hanzi2pinyin_split(city)
	date = pd.Timestamp(date).strftime("%Y-%m-%d")
    	res = weather[(weather['city'] == cityName) & (weather['date'] == date)]
    	weaHeader = ['lowest_temp','highest_temp','weather']
    	weaRes = {}
    	for i in range(0,len(weaHeader)):
	    weaRes[weaHeader[i]] = res.to_dict()[weaHeader[i]].values()[0]
    	return weaRes

def getLowT(weather,city,date):
	weaRes = getWeather(weather,city,date)
	return weaRes['lowest_temp']

def getHighT(weather,city,date):
	weaRes = getWeather(weather,city,date)
	return weaRes['highest_temp']

def getSunny(weather,city,date):
	weaRes = getWeather(weather,city,date)
	return weaRes['weather']

def getCityName(shop,shop_id):
	loc = shop['shop_id'].tolist().index(shop_id)
	return shop.loc[loc]['city_name']

def extractFeature(inPath,outPath,holiday,weather,shop,realStartTime):
	shop_id = int(inPath.split('/')[-1])
	cityName = getCityName(shop,shop_id)
	sortedData = pd.read_csv(inPath)
	
	#sortedData['time_stamp'] = pd.to_datetime(sortedData['time_stamp'])
	sortedData['time_stamp'] = sortedData['time_stamp'].astype(np.datetime64)
	m2Data = sortedData[sortedData['time_stamp'] >= pd.Timestamp(realStartTime)]
	m2Data.loc[:,'dayofweek'] = map(lambda x:x.dayofweek,m2Data['time_stamp'])
	m2Data.loc[:,'holiday'] = map(lambda x:getHoliday(holiday,x),m2Data['time_stamp'])
	
	#add weather feature
	m2Data.loc[:,'lt'] = map(lambda x:getLowT(weather,cityName,x),m2Data['time_stamp'])
	m2Data.loc[:,'ht'] = map(lambda x:getHighT(weather,cityName,x),m2Data['time_stamp'])
	m2Data.loc[:,'sunny'] = map(lambda x:getSunny(weather,cityName,x),m2Data['time_stamp'])
	
	m2Data.loc[:,'m2M'] = map(lambda x:m2M(sortedData,x),m2Data['time_stamp'])
	m2Data.loc[:,'w2A'] = map(lambda x:w2A(sortedData,x),m2Data['time_stamp'])
	m2Data.loc[:,'m1L'] = map(lambda x:m1L(sortedData,x),m2Data['time_stamp'])
	m2Data.loc[:,'d3w3'] = map(lambda x:d3w3(sortedData,x),m2Data['time_stamp'])
	m2Data.loc[:,'m1Lar'] = map(lambda x:m1Lar(sortedData,x),m2Data['time_stamp'])
	m2Data.loc[:,'w2Std'] = map(lambda x:w2Std(sortedData,x),m2Data['time_stamp'])

	f = open(outPath,'w')
	niceData = m2Data.loc[:,['time_stamp','dayofweek','holiday','lt','ht','sunny','m2M','w2A','m1L','d3w3','m1Lar','w2Std','pay_sum']]
	niceData.to_csv(f,index=False)
	f.close()

def iniParams():
	params = {}
	params['shopNum'] = 10
	# feature extracting for shop is from real start time.
	params['realStartTime'] = '2016-09-01'
	# outer data sources
	params['inPath'] = "../output-pay-sum/"
	params['holidayPath'] = "../holiday/holiday"
	params['weatherPath'] = "../input/weather.csv"
	params['shopPath'] = "../input/shop_info.txt"
	# path to save feature data for all shops
	params['outPath'] = "../output-feature-all/"
	params['shopHeader']=['shop_id','city_name','location_id','per_pay','score','comment_cnt','score',\
					'comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
	return params

def loadData(params):
	data = {}
	shop = pd.read_csv(params['shopPath'],header=None,names=params['shopHeader'])	
	weather = pd.read_csv(params['weatherPath'])
	holiday = pd.read_csv(params['holidayPath'])
	data['holiday'] = holiday
	data['weather'] = weather
	data['shop'] = shop
	return data

if __name__ == '__main__':
	
	params = iniParams()
	data = loadData(params)
	for i in range(1,params['shopNum']+1):
		print "shop_id:",i,",is processing..."
		extractFeature(params['inPath']+str(i),params['outPath']+str(i),data['holiday'],data['weather'],data['shop'],params['realStartTime'])	
		print "over..."

