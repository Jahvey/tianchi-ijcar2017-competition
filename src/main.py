'''
	desc: training and predict with aggregation
	author: zhpmatrix
'''
import csv
import operator
import numpy as np
import pandas as pd
from feature import *
import xgboost as xgb
import threading
#import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

from multiprocessing import Process,Queue,Pool,Manager
#from multiprocessing.dummy import Process,Queue,Pool,Manager

def frange(start,end,num):
	''' Generate data that's with the type of float'''
	return [start + float(i)*(end - start)/(float(num)-1) for i in range(num)]

def get_xgbr(data,X,y,params):
	# feature selection
	if params['features']['xgbr'] == 1:
		data,selectedFeatures = feature_selection(data,params['target'])
	else:
		selectedFeatures = params['cols']
	y = data.loc[:,params['target']].as_matrix()
	X = data.loc[:,params['cols']].as_matrix()
	# model selection
	if params['params']['xgbr'] == 1:
		# cross validation
		kf = KFold(n_splits=params['kfolds'],shuffle=True,random_state=params['seed'])
		# setting range of params
		depth_l = 3
		depth_h = 10
		estimators_l = 100
		estimators_h = 150
		# cross validation
		cross_errors = []
		for i in range(depth_l,depth_h):
			for j in range(estimators_l,estimators_h):
				errors = []
				for train_index,test_index in kf.split(X):
					xgbr=xgb.XGBRegressor(max_depth=i,n_estimators=j,nthread=4,silent=True)
					xgbr.fit(X[train_index],y[train_index].ravel())
					predictions = xgbr.predict(X[test_index])
					actuals = y[test_index]
					errors.append( mean_squared_error(actuals,predictions) )
				if params['verbose_params_sel'] == 1:
					print 'MAX_DEPTH=',i,',N_ESTIMATORS=',j,',CROSS_ERRORS=',avg(errors)
				cross_errors.append([i,j,avg(errors)])
				
		# choose base params
		params = sorted(cross_errors,key=lambda err:err[2],reverse=False)
		_max_depth = params[0][0]
		_n_estimators = params[0][1]

		print params,_max_depth,_n_estimators
		exit()
		# using params to construct model
		xgbr=xgb.XGBRegressor(max_depth=_max_depth,n_estimators=_n_estimators,silent=True)
		
		# fitting data
		xgbr.fit(X,y.ravel())
	else:
		_max_depth = 4
		_n_estimators = 118

		# using params to construct model
		xgbr=xgb.XGBRegressor(max_depth=_max_depth,n_estimators=_n_estimators,silent=True)
		
		# fitting data
		xgbr.fit(X,y.ravel())

	return xgbr,selectedFeatures

def get_gbr(data,X,y,params):
	selectedFeatures = params['cols']

	#model params train
	if params['params']['gbr'] == 1:
		#cross validation
		kf = KFold(n_splits=params['kfolds'], shuffle=True, random_state=params['seed'])
		#setting range of params
		depth_l = 3
		depth_h = 10
		estimators_l = 100
		estimators_h = 1100
		estimators_step = 350
		#cross validation
		cross_errors = []
		for i in range(depth_l,depth_h):
			for j in range(estimators_l,estimators_h,estimators_step):

				errors = []
				for train_index, test_index in kf.split(X):
					gbr = GradientBoostingRegressor(max_depth=i, n_estimators=j,random_state=params['seed'],verbose=False)
					gbr.fit(X[train_index], y[train_index].ravel())
					predictions = gbr.predict(X[test_index])
					actuals = y[test_index]
					errors.append(mean_squared_error(actuals, predictions))
				if params['verbose_params_sel'] == 1:
					print 'max_depth=', i, ',n_estimators=', j, ',CROSS_ERRORS=', avg(errors)
				cross_errors.append([i, j, avg(errors)])

		# choose best params
		params = sorted(cross_errors, key=lambda err: err[2], reverse=False)
		_max_depth = params[0][0]
		_estimators = params[0][1]

		print params, _max_depth, _estimators
		exit()
		# using params to construct model
		gbr = GradientBoostingRegressor(max_depth =_max_depth, n_estimators=_estimators,random_state=params['seed'])

		# fitting data
		gbr.fit(X, y.ravel())
	else:
		# loss = {'ls'}
		_max_depth = 7
		_estimators = 800
		gbr = GradientBoostingRegressor(max_depth=_max_depth, n_estimators=_estimators,random_state=params['seed'])

		# fitting data
		gbr.fit(X,y.ravel())
	return gbr,selectedFeatures


def get_svr(data,X,y,params):
	selectedFeatures = params['cols']
	
	if params['svr_scaler'] == 1:
		# scaling to 0-1
		minmax_scaler = preprocessing.MinMaxScaler()
		X = minmax_scaler.fit_transform(X)
	
	# model selection
	if params['params']['svr'] == 1:
		# cross validation
		kf = KFold(n_splits=params['kfolds'],shuffle=True,random_state=params['seed'])
		# setting range of params
		c_l = 0.1
		c_h = 2.0
		c_num = 20
		epsilon_l = 0.01
		epsilon_h = 0.30
		epsilon_num = 30
		_kernel = 'rbf'
		# cross validation
		cross_errors = []
		for i in frange(c_l,c_h,c_num):
			for j in frange(epsilon_l,epsilon_h,epsilon_num):
				errors = []
				for train_index,test_index in kf.split(X):
					svr = SVR(C=i,epsilon=j,kernel=_kernel,verbose=False)
                    			svr.fit(X[train_index],y[train_index].ravel())
					predictions = svr.predict(X[test_index])
					actuals = y[test_index]
					errors.append( mean_squared_error(actuals,predictions) )
				if params['verbose_params_sel'] == 1:
					print 'C=',i,',EPSILON=',j,',CROSS_ERRORS=',avg(errors)
				cross_errors.append([i,j,avg(errors)])
				
		# choose base params
		params = sorted(cross_errors,key=lambda err:err[2],reverse=False)
		_c = params[0][0]
		_epsilon  = params[0][1]

		print params,_c,_epsilon
		exit()
		# using params to construct model
		svr=SVR(C=_c,epsilon=_epsilon,kernel=_kernel)
		
		# fitting data
		svr.fit(X,y.ravel())
	else:
		_kernel = 'rbf'
		_c = 0.3
		_epsilon = 0.23
		svr = SVR(C=_c,epsilon=_epsilon,kernel=_kernel)
		# fitting data
        	svr.fit(X,y.ravel())
		
		#svr.fit(X,y.ravel())
	return svr,selectedFeatures

def get_rfr(data,X,y,params):
	selectedFeatures = params['cols']
	# model selection
	if params['params']['rfr'] == 1:
		# cross validation
		kf = KFold(n_splits=params['kfolds'],shuffle=True,random_state=params['seed'])
		# setting range of params
		estimators_l = 5
		estimators_h = 20
		# cross validation
		cross_errors = []
		for j in range(estimators_l,estimators_h):
			errors = []
			for train_index,test_index in kf.split(X):
				rfr=RandomForestRegressor(n_estimators=j,random_state=params['seed'],verbose=False)
				rfr.fit(X[train_index],y[train_index].ravel())
				predictions = rfr.predict(X[test_index])
				actuals = y[test_index]
				errors.append( mean_squared_error(actuals,predictions) )
			if params['verbose_params_sel'] == 1:
				print 'N_ESTIMATORS=',j,',CROSS_ERRORS=',avg(errors)
			cross_errors.append([j,avg(errors)])
			
		# choose base params
		params = sorted(cross_errors,key=lambda err:err[1],reverse=False)
		_n_estimators = params[0][0]

		print params,_n_estimators
		exit()
		# using params to construct model
		rfr=RandomForestRegressor(n_estimators=_n_estimators,random_state=params['seed'])
		
		# fitting data
		rfr.fit(X,y.ravel())
	else:
		_n_estimators = 12

		# using params to construct model
		rfr=RandomForestRegressor(n_estimators=_n_estimators,random_state=params['seed'])
		
		# fitting data
		rfr.fit(X,y.ravel())

	return rfr,selectedFeatures
	
def get_etr(data,X,y,params):
	selectedFeatures = params['cols']
	# model selection
	if params['params']['etr'] == 1:
		# cross validation
		kf = KFold(n_splits=params['kfolds'],shuffle=True,random_state=params['seed'])
		# setting range of params
		depth_l = 1
		depth_h = 20
		# cross validation
		cross_errors = []
		for j in range(depth_l,depth_h):
			errors = []
			for train_index,test_index in kf.split(X):
				etr=ExtraTreeRegressor(max_depth=j,random_state=params['seed'])
				etr.fit(X[train_index],y[train_index].ravel())
				predictions = etr.predict(X[test_index])
				actuals = y[test_index]
				errors.append( mean_squared_error(actuals,predictions) )
			if params['verbose_params_sel'] == 1:
				print 'MAX_DEPTH=',j,',CROSS_ERRORS=',avg(errors)
			cross_errors.append([j,avg(errors)])
			
		# choose base params
		params = sorted(cross_errors,key=lambda err:err[1],reverse=False)
		_max_depth = params[0][0]

		print params,_max_depth
		exit()
		# using params to construct model
		etr=ExtraTreeRegressor(max_depth=_max_depth,random_state=params['seed'])
		
		# fitting data
		etr.fit(X,y.ravel())
	else:
		_max_depth = 2

		# using params to construct model
		etr=ExtraTreeRegressor(max_depth=_max_depth,random_state=params['seed'])
		# fitting data
		etr.fit(X,y.ravel())

	return etr,selectedFeatures
def getRSet(data,X,y,params):
	'''
		desc: get model sets including XGBR,GBR,SVR,RFR
		param:
			X: training data
			y: target data
			kfolds: cross validation
			seed: seed for cross validation
		return: list,set of regressors
	'''
	
	regressors = []
	X = np.array(pd.DataFrame(X))
	y = np.array(pd.DataFrame(y))
	if params['models']['xgbr'] == 1:
		xgbr,xgbr_selectedFeatures = get_xgbr(data,X,y,params)
		regressors.append([xgbr,xgbr_selectedFeatures])
	if params['models']['gbr'] == 1:
		gbr,gbr_selectedFeatures = get_gbr(data,X,y,params)
		regressors.append([gbr,gbr_selectedFeatures])
	if params['models']['svr'] == 1:
		svr,svr_selectedFeatures = get_svr(data,X,y,params)
		regressors.append([svr,svr_selectedFeatures])
	if params['models']['rfr'] == 1:
		rfr,rfr_selectedFeatures = get_rfr(data,X,y,params)
		regressors.append([rfr,rfr_selectedFeatures])
	if params['models']['etr'] == 1:
		etr,etr_selectedFeatures = get_etr(data,X,y,params)
		regressors.append([etr,etr_selectedFeatures])
	return regressors

def loadData(filePath,params):
	'''
		desc: load data
		param:
			filePath: path of file to read
			cols: list,chosen features to train
			target: list,target column
		return:
			data: ndarray,origin data
			X: ndarray,training data
			y: ndarray,target data
	'''
	data = pd.read_csv(filePath)
	y = data.loc[:,params['target']].as_matrix()
	X = data.loc[:,params['cols']].as_matrix()
	return data,X,y

def iniParams():
	'''
		desc: initialize params including random seed and model weights
		return: dict,params
	'''
	params = {}
	params['kfolds'] = 10
	# number of training data,from 08-01(92) or 09-01(61)
	params['train_num'] = 92
	# whether to scale data for svr or not
	params['svr_scaler'] = 1
	# whether to tune paramters for specific model
	params['params'] = {'xgbr':0,'gbr':0,'svr':0,'rfr':0,'etr':0}
	# whether to do feature selection
	params['features'] = {'xgbr':1,'gbr':0,'svr':0,'rfr':0,'etr':0}
	# set weights for models,size of weights is the same as num of models
	params['weights'] = [1.0]
	#params['weights'] = [1.0,0.0,0.0,0.0,0.0]
	# whether to choose model or not
	params['models'] = {'xgbr':1,'gbr':0,'svr':0,'rfr':0,'etr':0}
	# whether to print msgs when tuning parameters
	params['verbose_params_sel'] = 1
	params['target'] = ['pay_sum']
	params['startTime'] = '2016-11-01'
	params['endTime'] = '2016-11-14'
	params['seed'] = 10
	params['cols'] = ['dayofweek','holiday','lt','ht','sunny','m2M','w2A','m1L','d3w3','m1Lar','w2Std']
	
	params['header'] = ['time_stamp','dayofweek','holiday','lt','ht','sunny','m2M','w2A','m1L','d3w3','m1Lar','w2Std','pay_sum']
	return params

def dirParams():
	params = {}
	params['curve'] = False
	params['shopNum'] = 2000
	params['train_num'] = 92
	params['preStartTime'] = "2016-11-01"
	params['submission'] = "../prediction.csv"
	params['inPath'] = "../output-feature-all/"
	params['resDir'] = "../output/"
	params['resImages'] = "../images/"
	params['holidayPath'] = "../holiday/holiday"
	params['weatherPath'] = "../input/weather.csv"
	params['shopPath'] = "../input/shop_info.txt"
	params['shopHeader']=['shop_id','city_name','location_id','per_pay','score','comment_cnt','score',\
					'comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
	return params
def worker(queue,start,_dirParams,step):
	import matplotlib.pyplot as plt
	fileDir = _dirParams['inPath']
	resDir = _dirParams['resDir']
	resImages = _dirParams['resImages']
	shopNum = _dirParams['shopNum']
	holiday = pd.read_csv(_dirParams['holidayPath'])
	weather = pd.read_csv(_dirParams['weatherPath'])
	shop = pd.read_csv(_dirParams['shopPath'],header=None,names=_dirParams['shopHeader'])
	_train_loss = queue.get()
	for i in range(start,start+step):
		print 'SHOP_ID=',i
		filePath = fileDir+str(i)
		figname = str(i)
		params = iniParams()
		data,X,y = loadData(filePath,params)
		data['time_stamp'] = pd.to_datetime(data['time_stamp'])
		regressors = getRSet(data,X,y,params)
		daysRange = pd.date_range(params['startTime'],params['endTime'])
		shop_id = i
		for i in range(0,len(daysRange)):
			data = insertDay(data,params['header'],daysRange[i])
			values,loc,start,end = getFeatureVals(data,params['cols'],holiday,weather,shop,shop_id,daysRange[i]) 	
			data.iloc[loc:loc+1,start:end+1] = [values]
			preVal = predictNew(X,regressors,values,params)
			data.iloc[loc:loc+1,end+1:end+2] = preVal			
		train_loss = showRes(plt,figname,resDir,resImages,X,data,regressors,params,_dirParams['curve'])
		_train_loss += train_loss
	queue.put(_train_loss)

def time_wrapper(func):
	def wrapper(*args,**kwargs):
		import time
		start = time.time()
		func(*args,**kwargs)
		end = time.time()
		print 'COST:{}'.format(end - start)
	return wrapper

@time_wrapper
def deal(_dirParams):
	shopNum = _dirParams['shopNum']
	train_num = _dirParams['train_num']
	step = 500
	_train_loss = 0.0
	manager = Manager()
	q = Queue()
	q.put(_train_loss)
	jobs = []
	for i in range(1,shopNum,step):
		p = Process(target=worker,args=(q,i,_dirParams,step))
		p.start()
		jobs.append(p)
	for p in jobs:
		p.join()
	print "_train_loss:",q.get() / (shopNum * train_num)
	submit(_dirParams)


def create_feature_map(features):
   	outfile = open('xgb.fmap', 'w')
    	i = 0
    	for feat in features:
       		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        	i = i + 1
    	outfile.close()
    
def feature_selection(data,target):
	'''
		desc: feature selection using xgboost
		params:
			data: origin data
			feaCols: columns to be erased
		return:
			data: new data with non-selected feature equals 0
			selectedFeatures: 
	'''
    features = [f for f in data.columns if f not in ['time_stamp','pay_sum']]
	y_train = data.pay_sum
	x_train = data[features]
	create_feature_map(features)

	num_rounds = 100
	xgb_params = {'silent':1}	 #don not pint msgs
	
	dtrain = xgb.DMatrix(x_train,label=y_train)
	model = xgb.train(xgb_params, dtrain, num_rounds,)

	importance = model.get_fscore(fmap='xgb.fmap')
	importance = sorted(importance.items(), key=operator.itemgetter(1))

	selectedFeatures = []
	allFeaturesNum = len(importance)
	selFeaturesNum = 5

	for i in range(allFeaturesNum,allFeaturesNum-selFeaturesNum,-1):
		selectedFeatures.append(importance[i-1][0])
	
	usedFeatures = list(set(target).union(set(selectedFeatures)))
	usedFeatures.append('time_stamp')
	notSelFeatures = [f for f in data.columns if f not in usedFeatures]
	data[notSelFeatures] = 0
	return data,selectedFeatures


def saveData(data,filePath):
	f = open(filePath,'w')
	data.to_csv(f,index=False)
	f.close()
	
def getLoss(realVals,preVals):
	'''
		desc: compute loss between the real values and predicting values,datetime range is [2016-10-18,2016-10-31]
		param:
			realVals: real values(column as pay_sum)
			preVals:  predicting values
	'''
	realVals = np.array(realVals.tolist())

	train_preVals = preVals
	train_realVals = realVals
	
	train_loss = (np.absolute(train_preVals - train_realVals.T)/np.absolute(train_preVals + train_realVals.T)).sum()
	
	return train_loss

def getFeatureVals(data,selFeatures,holiday,weather,shop,shop_id,time_stamp):
	'''
		desc: compute feature value given time_stamp
		param:
			data: origin data
			time_stamp: given time_stamp
		return:
			values: all feature values
			loc: location of the time_stamp in data
			
	'''
	start = 1       		# feature start index
	end = len(data.loc[0]) -1-1	# feature end index
	loc = list(data['time_stamp']).index(pd.Timestamp(time_stamp))
	cityName = getCityName(shop,shop_id)
	values = []
	values.append(time_stamp.dayofweek)
	values.append(getHoliday(holiday,time_stamp))
	values.append(getLowT(weather,cityName,time_stamp))
	values.append(getHighT(weather,cityName,time_stamp))
	values.append(getSunny(weather,cityName,time_stamp))
	values.append(m2M(data,time_stamp))
	values.append(w2A(data,time_stamp))
	values.append(m1L(data,time_stamp))
	values.append(d3w3(data,time_stamp))
	values.append(m1Lar(data,time_stamp))
	values.append(w2Std(data,time_stamp))
	return values,loc,start,end

def insertDay(data,header,time_stamp):
	insertDay = pd.DataFrame([[time_stamp,0,0,0,0,0,0,0,0,0,0,0,0]],columns=header)
	data = pd.concat([data,insertDay],ignore_index=True)
	return data
def get_sample_to_predict(selFeatures,sampleVals):
	sample= [0 for _ in range(0,11)]
	
	for fea in selFeatures:
		if fea == 'dayofweek':
			sample[0] = sampleVals[0]
		if fea == 'holiday':
			sample[1] = sampleVals[1]
		if fea == 'lt':
			sample[2] = sampleVals[2]
		if fea == 'ht':
			sample[3] = sampleVals[3]
		if fea == 'sunny':
			sample[4] = sampleVals[4]
		if fea == 'm2M':
			sample[5] = sampleVals[5]
		if fea == 'w2A':
			sample[6] = sampleVals[6]
		if fea == 'm1L':
			sample[7] = sampleVals[7]
		if fea  == 'd3w3':
			sample[8] = sampleVals[8]
		if fea  == 'm1Lar':
			sample[9] = sampleVals[9]
		if fea  == 'w2Std':
			sample[10] = sampleVals[10]
	return sample
def predictNew(X,regressors,sampleVals,params):
	'''
		desc:
		param:
			regressors:
			weights:
			sampleVals:
		return: double, weighted value
	'''
	weights = params['weights']
	preVals = []
	for reg in regressors:
		sample = get_sample_to_predict(reg[1],sampleVals)
		if params['svr_scaler'] == 1 and isinstance(reg[0],SVR) == True:
				_sample = sample[:]
				arr_sample = np.array([_sample])
				minmax_scaler = preprocessing.MinMaxScaler()
				scalerX = minmax_scaler.fit_transform(X)
				arr_sample = minmax_scaler.transform(arr_sample)
				_sample = arr_sample.tolist()[0]
				preVals.append(reg[0].predict(np.array(_sample).reshape(1,-1)))

		else:
				preVals.append(reg[0].predict(np.array(sample).reshape(1,-1)))	
	weights = np.array(weights)
	preVals = np.array(preVals)
	weights.shape = (len(weights),1)
	weights = np.transpose(weights)
	comVals = weights.dot(preVals)
	return comVals[0][0]

def showRes(plt,figname,resDir,resImages,X,data,regressors,params,curve):
	weights = params['weights']
	preVals = []
	for reg in regressors:
		# shallow copy
		_data = data.copy()

		# choosing data according to columns
		for param in params['cols']:
			if param not in reg[1]:
				_data[param] = 0
		
		newX = _data.loc[:,params['cols']].as_matrix()
		if params['svr_scaler'] == 1 and isinstance(reg[0],SVR) == True:
				_newX = newX[:]
				minmax_scaler = preprocessing.MinMaxScaler()
				scalerX = minmax_scaler.fit_transform(X)
				_newX = minmax_scaler.transform(_newX)
				preVals.append( reg[0].predict(_newX) )
		else:
				preVals.append( reg[0].predict(newX))
	weights = np.array(weights)
	preVals = np.array(preVals)
	weights.shape = (len(weights),1)
	weights = np.transpose(weights)
	comVals = weights.dot(preVals)
	data['predict']=comVals.tolist()[0]
	train_loss = getLoss(data['pay_sum'],comVals)
	label = ['pay_sum','predict']
	labelColor = ['red','green']
	for i in range(0,len(label)):
		plt.plot(data[label[i]],color=labelColor[i],label=label[i])
	plt.legend(loc='best')
	plt.title(figname)
	plt.savefig(resImages+str(figname))
	if curve == True:
		plt.show()
	plt.close()
	saveData(data,resDir+figname)
	return train_loss

def submit(params):
	f = open(params['submission'],'wb')
	wr = csv.writer(f)
	for i in range(1,params['shopNum']+1):
		data = pd.read_csv(params['resDir']+str(i))
		shopPre = []
		shopPre.append(i)
		start = data['time_stamp'].tolist().index(params['preStartTime'])
		for j in range(start,start+14):
			shopPre.append(int(data.loc[j]['predict']))
		newShopPre = []
		for i in range(1,len(shopPre)):
			if shopPre[i] > 0:
				newShopPre.append(shopPre[i])
		
		avgShopPre = sum(newShopPre)/len(newShopPre)
		for i in range(0,len(shopPre)):
			if shopPre[i] < 0:
				shopPre[i] = avgShopPre
		wr.writerow(shopPre)
	f.close()


if __name__ == '__main__':
	_dirParams = dirParams()
	deal(_dirParams)
	
