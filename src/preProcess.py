#
#	desc: compute pay sum and get holiday states at every day for each shop
#	author: zhpmatrix@datarush
#	date: 2017-01-18

from pinyin import *
import pandas as pd
import urllib, urllib2

def get_day_type(time_stamp):
	'''
		desc: return time_stamp is whether holiday or not
			holiday and rest day: 1
			workday: 0
		author: summer
	'''
	url = 'http://www.easybots.cn/api/holiday.php?d='+time_stamp
    	req = urllib2.Request(url)
    	resp = urllib2.urlopen(req)
    	content = resp.read()
   	if (content):
        	day_type = content[content.rfind(":")+2:content.rfind('"')]
	return 1 if day_type == '2' or day_type == '1'  else 0


def groupFile(inPath,outPath,shopNum):
	userData = pd.read_csv(inPath,header=None,names=['user_id','shop_id','time_stamp']);
	userData['time_stamp'] = map(lambda x: x.strftime('%Y-%m-%d'), pd.to_datetime(userData['time_stamp']))
	groupData = pd.DataFrame(userData.groupby(['shop_id','time_stamp']).size().reset_index(name='pay_sum'))
	for i in range(1,shopNum+1):
		f = open(outPath+str(i),'w')
		groupData[groupData['shop_id']==i].to_csv(f,header=True,index=False,columns=['time_stamp','pay_sum'])
		f.close()
		
def checkHoliday(startTime,endTime,filePath):
	dates = pd.date_range(startTime,endTime)
	
	print "request holiday from easybots..."
	dayType = [get_day_type(date.strftime("%Y-%m-%d")) for date in dates] 
	print "request over..."

	holiday = pd.DataFrame({'time_stamp':dates,'holiday':dayType})
	f = open(filePath,'w')
	holiday.to_csv(f,index=False)
	f.close()

def deal():
	inPath = "../input/user_pay.txt"
	outPath = "../output-pay-sum/"
	shopNum = 2000
	print "start grouping data to get pay_sum for each shop..."
	groupData = groupFile(inPath,outPath,shopNum)
	print "end grouping..."

	holidayPath = "../holiday/holiday"
	startTime = '20160901'
	endTime = '20161231'
	print "string check holiday for each day..."
	checkHoliday(startTime,endTime,holidayPath)
	print "end checking holiday..."

if __name__ == "__main__":
	deal()
