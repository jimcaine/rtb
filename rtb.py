import csv
import random
import numpy as np
import pandas as pd
from sklearn import linear_model, tree, lda, naive_bayes
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


CHARTS_DIR_PATH = './charts/'
TABLES_DIR_PATH = './tables/'
DATA_PATH = './data/'
IMP_TRAIN_PATH = './raw_data/imp_train.txt'
CLICK_TRAIN_PATH = './raw_data/click_train.txt'
IMP_TEST_PATH = './raw_data/imp_test.txt'
CLICK_TEST_PATH = './raw_data/click_test.txt'


################################################################################
# DATA PREPROCESSING
################################################################################
def feature_selection_imps(file_path):
	'''
	This method will take a txt file as input and creates a CSV file n
	Output: imps_featureselection.csv
	'''
	matrix = []
	header = ['bid_id',
			  'ipinyou_id',
			  'timestamp',
			  'hour',
			  'browser_chrome',
			  'browser_ie',
			  'browser_safari',
			  'browser_firefox',
			  'mobile',
			  'iphone',
			  'ipad',
			  'android',
			  'windows',
			  'linux',
			  'region_id',
			  'ad_exchange',
			  'domain',
			  'ad_slot_id',
			  'ad_slot_size',
			  'ad_slot_visibility',
			  'ad_slot',
			  'ad_slot_floor_price',
			  'paying_price']
	matrix.append(header)	

	with open(file_path) as f:
		csvreader = csv.reader(f, delimiter='\t')
		for row in csvreader:
			# label the attributes
			bid_id = row[0]
			timestamp = row[1]
			log_type = row[2]
			ipinyou_id = row[3]
			useragent = row[4]
			ip_address = row[5]
			region_id = row[6]
			city_id = row[7]
			ad_exchange = row[8]
			domain = row[9]
			url = row[10]
			anonymous_url_id = row[11]
			ad_slot_id = row[12]
			ad_slot_width = row[13]
			ad_slot_height = row[14]
			ad_slot_visibility = row[15]
			ad_slot = row[16]
			ad_slot_floor_price = row[17]
			creative_id = row[18]
			bidding_price = row[19]
			paying_price = row[20]
			key_page_url = row[21]
			advertiser_id = row[22]
			user_tags = row[23]

			# create derivative attributes
			year = timestamp[0:5]
			month = timestamp[5:7]
			day = timestamp[7:9]
			hour = timestamp[8:10]
			minute = timestamp[10:12]

			if 'Chome' in useragent:
				browser_chrome = 1
			else:
				browser_chrome = 0

			if 'MSIE' in useragent:
				browser_ie = 1
			else:
				browser_ie = 0

			if 'Safari' in useragent:
				browser_safari = 1
			else:
				browser_safari = 0

			if 'Firefox' in useragent:
				browser_firefox = 1
			else:
				browser_firefox = 0
			
			if 'Mobile' in useragent:
				mobile = 1
			else:
				mobile = 0

			if 'iPhone' in useragent:
				iphone = 1
			else:
				iphone = 0

			if 'iPad' in useragent:
				ipad = 1
			else:
				ipad = 0

			if 'Android' in useragent:
				android = 1
			else:
				android = 0

			if 'Windows NT' in useragent:
				windows = 1
			else:
				windows = 0

			if 'Linux' in useragent:
				linux = 1
			else:
				linux = 0

			ad_slot_size = str(ad_slot_width)+'x'+str(ad_slot_height)

			vector = [bid_id,
					  ipinyou_id,
					  timestamp,
					  hour,
					  browser_chrome,
					  browser_ie,
					  browser_safari,
					  browser_firefox,
					  mobile,
					  iphone,
					  ipad,
					  android,
					  windows,
					  linux,
					  region_id,
					  ad_exchange,
					  domain,
					  ad_slot_id,
					  ad_slot_size,
					  ad_slot_visibility,
					  ad_slot,
					  ad_slot_floor_price,
					  paying_price]
			matrix.append(vector)

	df = pd.DataFrame(matrix)
	df.columns = matrix[0]
	df = df.ix[1:]

	return df


def feature_selection_clicks(file_path):
	matrix = []
	header = ['bid_id',
			  'ipinyou_id',
			  'timestamp']
	matrix.append(header)

	with open(file_path) as f:
		csvreader = csv.reader(f, delimiter='\t')
		for row in csvreader:
			# label the attributes
			bid_id = row[0]
			timestamp = row[1]
			ipinyou_id = row[3]

			vector = [bid_id,
					  ipinyou_id,
					  timestamp]
			matrix.append(vector)

	df = pd.DataFrame(matrix)
	df.columns = matrix[0]
	df = df.ix[1:]

	return df


def merge_impression_click(imps, clicks):
	imps = imps.set_index(imps['bid_id'])
	imps = imps.drop(['bid_id'], axis=1)
	
	clicks = clicks.set_index(clicks['bid_id'])
	clicks = clicks.drop(['bid_id'], axis=1)

	click_indicator = []
	for index, value in imps.iterrows():
		# find the timestamp and userid
		timestamp = value['timestamp']
		userid = value['ipinyou_id']
		# if the userid and (timestamp + 5 minutes) is in the click log
		if userid in list(clicks['ipinyou_id']):
			click_indicator.append(1)
		else:
			click_indicator.append(0)
	imps['clicks'] = click_indicator
	
	return imps


def transform_ssf(df):
	'''
	Transforms categorical variables to dummy variables
	'''
	df = df.set_index(df['timestamp'])

	hour_dummies = pd.get_dummies(df['hour'])
	hour_dummies_columns = []
	for v in hour_dummies.columns.values:
		hour_dummies_columns.append('hour_%s' % str(v))
	hour_dummies.columns = hour_dummies_columns	

	region_dummies = pd.get_dummies(df['region_id'])
	region_dummies_columns = []
	for v in region_dummies.columns.values:
		region_dummies_columns.append('region_%s' % str(v))
	region_dummies.columns = region_dummies_columns

	adexchange_dummies = pd.get_dummies(df['ad_exchange'])
	adexchange_dummies_columns = []
	for v in adexchange_dummies.columns.values:
		adexchange_dummies_columns.append('exchange_%s' % str(v))
	adexchange_dummies.columns = adexchange_dummies_columns

	# domain_dummies = pd.get_dummies(df['domain'])
	# domain_dummies_columns = []
	# for v in domain_dummies.columns.values:
	# 	domain_dummies_columns.append('domain_%s' % str(v))
	# domain_dummies.columns = domain_dummies_columns

	adslotsize_dummies = pd.get_dummies(df['ad_slot_size'])
	adslotsize_dummies_columns = []
	for v in adslotsize_dummies.columns.values:
		adslotsize_dummies_columns.append('adslotsize_%s' % str(v))
	adslotsize_dummies.columns = adslotsize_dummies_columns

	adslotvisibility_dummies = pd.get_dummies(df['ad_slot_visibility'])
	adslotvisibility_dummies_columns = []
	for v in adslotvisibility_dummies.columns.values:
		adslotvisibility_dummies_columns.append('adslotvis_%s' % str(v))
	adslotvisibility_dummies.columns = adslotvisibility_dummies_columns

	adslot_dummies = pd.get_dummies(df['ad_slot'])
	adslot_dummies_columns = []
	for v in adslot_dummies.columns.values:
		adslot_dummies_columns.append('adslot_%s' % str(v))
	adslot_dummies.columns = adslot_dummies_columns

	df = df.drop(['ipinyou_id',
				  'timestamp',
				  'hour',
				  'region_id',
				  'ad_exchange',
				  'domain',
				  'ad_slot_id',
				  'ad_slot_size',
				  'ad_slot_visibility',
				  'ad_slot',
				  'ad_slot_floor_price'], axis=1)
	df_ssf = pd.concat([df,
						hour_dummies,
						region_dummies,
						adexchange_dummies,
						adslotsize_dummies,
						adslotvisibility_dummies,
						adslot_dummies], axis=1)

	return df_ssf


def sample_train_data(df, perc_no_clicks=500):
	df = df.reset_index()
	df_clicks = df[df['clicks'] == 1]
	df_noclicks = df[df['clicks'] == 0]
	df_noclicks_sample_rows = \
		random.sample(df_noclicks.index, int(len(df_clicks)*(perc_no_clicks/100)))
	df_noclicks_sample = df_noclicks.ix[df_noclicks_sample_rows]
	df = pd.concat([df_clicks, df_noclicks_sample])
	df = df.set_index('timestamp')
	return df


def partition(df):
	df = df.sort_index()
	df_length = df.shape[0]
	train = df.iloc[0:int(0.8*df_length)]
	test = df.iloc[(int(0.8*df_length)+1):]

	return train, test




################################################################################
# DATA EXPLORATION
################################################################################
def data_exploration():
	# create data frame from raw data
	# file_path = './data/impression_sample.txt'
	file_path = './raw_data/impression_sample.txt'
	matrix = []
	header = ['bid_id',
			  'year',
			  'day',
			  'hour',
			  'minute',
			  'log_type',
			  'ipinyou_id',
			  'browser_chrome',
			  'browser_ie',
			  'browser_safari',
			  'browser_firefox',
			  'browser_other',
			  'mobile',
			  'iphone',
			  'ipad',
			  'android',
			  'windows',
			  'linux',
			  'region_id',
			  'city_id',
			  'ad_exchange',
			  'domain',
			  'url',
			  'ad_slot_id',
			  'ad_slot_size',
			  'ad_slot_visibility',
			  'ad_slot',
			  'ad_slot_floor_price',
			  'creative_id',
			  'key_page_url',
			  'advertiser_id',
			  'user_tags']
	matrix.append(header)

	with open(file_path) as f:
		csvreader = csv.reader(f, delimiter='\t')
		for row in csvreader:
			# label the attributes
			bid_id = row[0]
			timestamp = row[1]
			log_type = row[2]
			ipinyou_id = row[3]
			useragent = row[4]
			ip_address = row[5]
			region_id = row[6]
			city_id = row[7]
			ad_exchange = row[8]
			domain = row[9]
			url = row[10]
			anonymous_url_id = row[11]
			ad_slot_id = row[12]
			ad_slot_width = row[13]
			ad_slot_height = row[14]
			ad_slot_visibility = row[15]
			ad_slot = row[16]
			ad_slot_floor_price = row[17]
			creative_id = row[18]
			bidding_price = row[19]
			paying_price = row[20]
			key_page_url = row[21]
			advertiser_id = row[22]
			user_tags = row[23]

			# create derivative attributes
			year = timestamp[0:4]
			month = timestamp[4:6]
			day = timestamp[6:8]
			hour = timestamp[8:10]
			minute = timestamp[10:12]

			if 'Chome' in useragent:
				browser_chrome = 1
			else:
				browser_chrome = 0

			if 'MSIE' in useragent:
				browser_ie = 1
			else:
				browser_ie = 0

			if 'Safari' in useragent:
				browser_safari = 1
			else:
				browser_safari = 0

			if 'Firefox' in useragent:
				browser_firefox = 1
			else:
				browser_firefox = 0

			if 'Chrome' not in useragent \
			and 'MSIE' not in useragent \
			and 'Safari' not in useragent \
			and 'Firefox' not in useragent:
				browser_other = 1
			else:
				browser_other = 0
			
			if 'Mobile' in useragent:
				mobile = 1
			else:
				mobile = 0

			if 'iPhone' in useragent:
				iphone = 1
			else:
				iphone = 0

			if 'iPad' in useragent:
				ipad = 1
			else:
				ipad = 0

			if 'Android' in useragent:
				android = 1
			else:
				android = 0

			if 'Windows NT' in useragent:
				windows = 1
			else:
				windows = 0

			if 'Linux' in useragent:
				linux = 1
			else:
				linux = 0

			ad_slot_size = str(ad_slot_width) + 'x' + str(ad_slot_height)

			vector = [bid_id,
					  year,
					  day,
					  hour,
					  minute,
					  log_type,
					  ipinyou_id,
					  browser_chrome,
					  browser_ie,
					  browser_safari,
					  browser_firefox,
					  browser_other,
					  mobile,
					  iphone,
					  ipad,
					  android,
					  windows,
					  linux,
					  region_id,
					  city_id,
					  ad_exchange,
					  domain,
					  url,
					  ad_slot_id,
					  ad_slot_size,
					  ad_slot_visibility,
					  ad_slot,
					  ad_slot_floor_price,
					  creative_id,
					  key_page_url,
					  advertiser_id,
					  user_tags]
			matrix.append(vector)


	# turn matrix into pandas dataframe
	df = pd.DataFrame(matrix)
	df.columns = matrix[0]
	df = df.ix[1:]
	df.index = df['bid_id']
	df = df.drop(['bid_id'], axis=1)


	def histogram(feature):
		vc = df[feature].value_counts()
		freq = vc.values

		plt.hist(freq)
		plt.savefig(CHARTS_DIR_PATH + ('de-%s-hist.png' % feature))
		plt.clf()


	def piechart(feature):
		vc = df[feature].value_counts()
		vc = vc.sort_index()

		labels = vc.index.values
		fracs = vc.values
		plt.pie(fracs, labels=labels, autopct='%1.1f%%')
		plt.savefig(CHARTS_DIR_PATH + ('de_%s-pie.png' % feature))
		plt.clf()


	# TIMESTAMP
	events_per_hour = df['hour'].value_counts()
	events_per_hour = events_per_hour.sort_index()

	events_per_hour.plot()
	plt.savefig(CHARTS_DIR_PATH + 'de_events-per-hour.png')
	plt.clf()


	# FREQUENCY
	histogram('ipinyou_id')


	# BROWSER
	num_firefox = df['browser_firefox'].sum()
	num_safari = df['browser_safari'].sum()
	num_ie = df['browser_ie'].sum()
	num_chrome = df['browser_chrome'].sum()
	num_other = df['browser_other'].sum()

	ind = np.array([0,1,2,3,4])
	width = 0.75
	fig, ax = plt.subplots()
	rects = ax.bar(left=ind,
				   height=[num_firefox, num_safari, num_ie, num_chrome, num_other],
				   width=width)
	ax.set_xticks(ind + (width/2))
	ax.set_xticklabels(['Firefox', 'Safari', 'IE', 'Chrome', 'Other'])
	plt.savefig(CHARTS_DIR_PATH + 'de_browser-distribution.png')
	plt.clf()


	# MOBILE VS NONMOBILE
	num_mobile = df['mobile'].sum()
	num_nonmobile = df['mobile'].count() - num_mobile

	labels = 'Mobile', 'Non-mobile'
	fracs = [num_mobile, num_nonmobile]
	plt.pie(fracs, labels=labels)
	plt.savefig(CHARTS_DIR_PATH + 'de_mobile-vs-nonmobile.png')
	plt.clf()


	# REGIONS AND CITIES
	num_regions = len(df['region_id'].unique())
	num_cities = len(df['city_id'].unique())
	print '### REGIONS AND CITIES ###'
	print 'Number of unique regions: %s' % str(num_regions)
	print 'Number of unique cities: %s' % str(num_cities)
	print
	print '--'

	# AD EXCHANGE ID
	piechart('ad_exchange')

	# DOMAIN
	histogram('domain')

	# URL
	histogram('url')

	# AD SLOT ID
	histogram('ad_slot_id')

	# AD SLOT SIZE
	piechart('ad_slot_size')

	# AD SLOT VISIBILITY
	piechart('ad_slot_visibility')

	# AD SLOT
	piechart('ad_slot')

	# AD SLOT FLOOR PRICE
	histogram('ad_slot_floor_price')

	# CREATIVE ID
	print df['creative_id'].value_counts()

	# KEY PAGE URL

	# ADVERTISER ID

	# USER TAGS

	print 




################################################################################
# MODELS AND MODEL EVALUATION
################################################################################
def fit_model(df, model='logistic_regression', max_depth=5):
	# split the target attribute from the explanatory variables
	y = df['clicks']
	x = df.drop(['clicks', 'paying_price'], axis=1)
	
	# fit the model
	if model == 'logistic_regression':
		m = linear_model.LogisticRegression()
		m.fit(x, y)
		df_coef = pd.DataFrame(columns=['attribute','logreg_coef'])
		df_coef['attribute'] = x.columns.values
		df_coef['logreg_coef'] = np.ravel(m.coef_)
		df_coef.to_csv(TABLES_DIR_PATH + 'logreg_coef.csv')
		print 'Logistic regression coefficients:'
		print df_coef
	elif model == 'tree':
		m = tree.DecisionTreeClassifier(max_depth=max_depth)
		m.fit(x, y)
	elif model == 'lda':
		m = lda.LDA()
		m.fit(x, y)
	elif model == 'naivebayes':
		m = naive_bayes.GaussianNB()
		m.fit(x, y)

	return m


def create_propensity_histogram(df_test, model):
	# keep a running total of the propensities
	propensities_list = []

	# for each impression in the log
	for index, value in df_test.iterrows():
		# grab the attributes of the impression (drop clicks)
		attributes = value
		attributes = attributes.drop(['clicks', 'paying_price'])

		# calculate the impression utility and append to propensities list
		imp_propensity = model.predict_proba(attributes)[0][1]
		propensities_list.append(imp_propensity)

	# append propensities to df_test
	df_test['propensity'] = propensities_list

	# create histogram of propensities
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(propensities_list)
	ax.set_xlabel('Propensity To Click')
	ax.set_ylabel('Frequency')
	if 'LogisticRegression' in str(type(model)):
		ax.set_title('Histogram of Propensity To Click (Logistic Regression)')
		plt.savefig(CHARTS_DIR_PATH + 'hist_prop_to_click_logreg.png')
	elif 'DecisionTreeClassifier' in str(type(model)):
		ax.set_title('Histogram of Propensity To Click (Decision Tree)')
		plt.savefig(CHARTS_DIR_PATH + 'hist_prop_to_click_tree.png')
	elif 'LDA' in str(type(model)):
		ax.set_title('Histogram of Propensity To Click (LDA)')
		plt.savefig(CHARTS_DIR_PATH + 'hist_prop_to_click_lda.png')
	elif 'GaussianNB' in str(type(model)):
		ax.set_title('Histogram of Propensity To Click (Naive Bayes)')
		plt.savefig(CHARTS_DIR_PATH + 'hist_prop_to_click_naivebayes.png')	
	plt.clf()


def evaluate_model_on_full_set(df_test, model):
	# for each impression in the log
	predictions = []
	propensities = []
	for index, value in df_test.iterrows():
		attributes = value
		attributes = attributes.drop(['clicks', 'paying_price'])
		propensity = model.predict_proba(attributes)[0][1]
		pred = model.predict(attributes)[0]
		propensities.append(propensity)
		predictions.append(pred)
	
	df_analysis = df_test.copy()
	df_analysis['propensity'] = propensities
	df_analysis['pred'] = predictions
	df_analysis = df_analysis.sort(columns=['propensity'], ascending=False)
	df_analysis['cum_click'] = df_analysis['clicks'].cumsum()
	df_analysis['cum_click_perc'] = \
		100*df_analysis['cum_click'] / df_analysis['clicks'].sum()
	ninety_perc_click_threshold = \
		df_analysis[df_analysis['cum_click_perc'] < 0.9].shape[0] / float(df_analysis.shape[0])

	cm = confusion_matrix(df_analysis['clicks'], df_analysis['pred'])
	accuracy = (cm[0][0]+cm[1][1]) / float(cm.sum())
	recall = cm[1][1] / float(cm[1][0] + cm[1][1])
	precision = cm[1][1] / float(cm[0][1] + cm[1][1])

	print cm

	return accuracy, recall, precision, cm, ninety_perc_click_threshold*100.0


################################################################################
# SIMULATE PERFORMANCE
################################################################################
def simulate_flat_bid(bid=300, print_output=True):
	# create df with all impressions
	df = pd.DataFrame.from_csv('./data/test_ssf_100.csv')

	# keep running total of ad spend, impressions, clicks
	ad_spend = 0
	impressions = 0
	clicks = 0

	# for each impression in the log
	for index, value in df.iterrows():
		# grab the attributes of the impression (drop clicks)
		attributes = value

		# calculate the impression utility
		imp_utility = bid

		# calculate the minimum utility needed
		min_utility = 0
		adjusted_utility = imp_utility - min_utility

		# calculate the bid price
		bid_price = adjusted_utility

		# find the paying price
		paying_price = value['paying_price']

		# if bid_price > paying_price, update campaign metrics
		if bid_price >= paying_price:
			impressions += 1
			ad_spend += paying_price/1000.0
			if value['clicks'] == 1:
				clicks += 1

	try:
		cpm = float(ad_spend) / (impressions) * 1000
		ctr = float(clicks) / impressions
	except ZeroDivisionError:
		cpm = 0.0
		ctr = 0.0
	try:
		cpc = float(ad_spend) / clicks
	except ZeroDivisionError:
		cpc = 0.0

	if print_output == False:
		return ad_spend, impressions, clicks, cpm, ctr, cpc
	else:
		print 'Total ad spend: %.2f Yen' % ad_spend
		print 'Number of impressions delivered: %d' % impressions
		print 'Number of clicks: %d' % clicks
		print 'eCPM: %.2f Yen' % cpm
		print 'CTR: %0.2f %%' % (ctr*100)
		print 'CPC: %.2f Yen' % cpc


def simulate_goal_bid(goal, model, print_output=True):
	# create df with all impressions
	df = pd.DataFrame.from_csv('./data/test_ssf_100.csv')

	# keep track of bids
	bids = []

	# for each impression in the log
	for index, value in df.iterrows():
		# grab the attributes of the impression (drop clicks)
		attributes = value
		attributes = attributes.drop(['clicks', 'paying_price'])

		# calculate the bid
		imp_utility = model.predict_proba(attributes)[0][1]
		bid = imp_utility*goal
		bids.append(bid)

	df['bid'] = bids
	df['impression'] = (df['bid'] > df['paying_price'])*1
	df_winning_bids = df[df['impression'] == 1]
	impressions = df_winning_bids.shape[0]
	clicks = df_winning_bids['clicks'].sum()
	ad_spend = df_winning_bids['paying_price'].sum()/1000.0

	cpm = float(ad_spend) / (impressions) * 1000
	ctr = float(clicks) / impressions
	cpc = float(ad_spend) / clicks

	if print_output == False:
		return ad_spend, impressions, clicks, cpm, ctr, cpc

	# create bid and paying_price histogram
	plt.hist(list(df['bid']), alpha=0.8, label='bid')
	plt.hist(list(df['paying_price']), alpha=0.5, label='actual')
	plt.legend(loc='upper right')
	plt.xlabel('Price')
	plt.ylabel('Frequency')
	plt.title('Bid and Price Distribution')
	plt.savefig(CHARTS_DIR_PATH + 'bid-distribution_goal_%s.png' % str(goal))

	# print output
	print 'Goal: %s' % str(goal)
	print 'Total ad spend: %.2f Yen' % ad_spend
	print 'Number of impressions delivered: %d' % impressions
	print 'Number of clicks: %d' % clicks
	print 'eCPM: %.2f Yen' % cpm
	print 'CTR: %0.2f %%' % (ctr*100)
	print 'CPC: %.2f Yen' % cpc
	print '-'*50

	return ad_spend, impressions, clicks, cpm, ctr, cpc


def calculate_spend_urgency(budget, df_imps, timestamp):
	# calculate the total worth of the bids left
	total_worth_of_bids_left = \
		df_imps[df_imps.index > timestamp]['paying_price'].sum()
	# print total_worth
	spend_urgency = (budget*1000.0 / total_worth_of_bids_left)*100.0
	if spend_urgency > 0:
		return spend_urgency
	else:
		return 0.0

		
def calculate_cpe_threshold(historical_bids, spend_urgency, model):
	# randomly sample historical bids
	bids_sample_idx = random.sample(historical_bids.index, 500)
	bids_sample = historical_bids.ix[bids_sample_idx]

	# calculate propensity to convert for all impressions
	propensities = []
	for index, value in bids_sample.iterrows():
		attributes = value.drop(['clicks', 'paying_price'])
		propensity = model.predict_proba(attributes)[0][1]
		propensities.append(propensity)
	bids_sample['propensity'] = propensities

	# calculate cpe for each bid
	bids_sample['cpe'] = bids_sample['paying_price'] / bids_sample['propensity']

	# sort the impressions be cpe
	bids_sample = bids_sample.sort(columns=['cpe'], ascending=True)

	# calculate the total ad spend in df
	total_ad_spend = bids_sample['paying_price'].sum()

	# create a column in df representing cumulative perc of ad spend
	bids_sample['cum_spend'] = bids_sample['paying_price'].cumsum()
	bids_sample['cum_spend_perc'] = \
		100*bids_sample['cum_spend'] / bids_sample['paying_price'].sum()
	cpe_threshold = \
		bids_sample[bids_sample['cum_spend_perc'] < spend_urgency].iloc[-1]['cpe']

	return cpe_threshold


def simulate_dynamic_bid(budget, model, print_output=True):
	budget0 = budget
	print 'Budget0: %s' % str(budget)
	# end date
	end_time = 20130609000110181

	# create df with all impressions
	df = pd.DataFrame.from_csv('./data/test_ssf_100.csv')

	# keep track of bids
	num_bids = 0
	impressions = 0
	clicks = 0
	bids = []

	# define starting cpe threshold
	cpe_threshold = 80
	# for each impression in the log
	for index, value in df.iterrows():
		# update max CPE
		num_bids += 1
		if 0.02 > float(budget) / budget0 > 0.95:
			if num_bids % 500 == 0:
				# create data frame of historical bids
				historical_bids = df[df.index <= index]

				# compute the spend urgency
				spend_urgency = calculate_spend_urgency(budget, df, index)

				# calculate new max CPE threshold
				if spend_urgency == 0:
					cpe_threshold = np.inf
				else:
					cpe_threshold = \
						calculate_cpe_threshold(historical_bids, spend_urgency, model)
		else:
			if num_bids % 5000 == 0:
				# create data frame of historical bids
				historical_bids = df[df.index <= index]

				# compute the spend urgency
				spend_urgency = calculate_spend_urgency(budget, df, index)

				# calculate new max CPE threshold
				if spend_urgency == 0:
					cpe_threshold = np.inf
				else:
					cpe_threshold = \
						calculate_cpe_threshold(historical_bids, spend_urgency, model)


		# grab the attributes of the impression (drop clicks)
		attributes = value
		price = attributes['paying_price']
		attributes = attributes.drop(['clicks', 'paying_price'])

		# calculate CPE
		propensity = model.predict_proba(attributes)[0][1]
		
		# calculate bid
		bid = propensity * cpe_threshold
		if budget < 0:
			bid = 0.0
		bids.append(bid)

		# update real time metrics
		if bid >= price:
			budget -= price/1000.0
			impressions += 1

	df['bid'] = bids
	df['impression'] = (df['bid'] > df['paying_price'])*1

	df_winning_bids = df[df['impression'] == 1]
	impressions = df_winning_bids.shape[0]
	clicks = df_winning_bids['clicks'].sum()
	ad_spend = df_winning_bids['paying_price'].sum()/1000.0

	cpm = float(ad_spend) / (impressions) * 1000.0
	ctr = float(clicks) / impressions
	cpc = float(ad_spend) / clicks

	if print_output == False:
		return ad_spend, impressions, clicks, cpm, ctr, cpc
	else:
		print 'Total ad spend: %.2f Yen' % ad_spend
		print 'Number of impressions delivered: %d' % impressions
		print 'Number of clicks: %d' % clicks
		print 'eCPM: %.2f Yen' % cpm
		print 'CTR: %0.2f %%' % (ctr*100)
		print 'CPC: %.2f Yen' % cpc
		print '-'*50
		return ad_spend, impressions, clicks, cpm, ctr, cpc




################################################################################
# MAIN
################################################################################
def evaluate_models(df_train, df_test):
	# logistic regression
	model_logreg = fit_model(df_train, model='logistic_regression')
	accuracy_unseen_logreg, recall_unseen_logreg, precision_unseen_logreg, \
		cm_unseen_logreg, ninety_perc_logreg = \
			evaluate_model_on_full_set(df_test, model_logreg)
	print '### LOGISTIC REGRESSION ###'
	print accuracy_unseen_logreg*100
	print recall_unseen_logreg*100
	print precision_unseen_logreg*100
	print ninety_perc_logreg
	print '-'*100

	# decision tree (max depth = 2)
	model_tree2 = fit_model(df_train, model='tree', max_depth=2)
	accuracy_unseen_tree2, recall_unseen_tree2, precision_unseen_tree2, \
		cm_unseen_tree2, ninety_perc_tree2 = \
			evaluate_model_on_full_set(df_test, model_tree2)
	print '### DECISION TREE (max_depth=2) ###'
	print accuracy_unseen_tree2*100
	print recall_unseen_tree2*100
	print precision_unseen_tree2*100
	print ninety_perc_tree2
	print '-'*100

	# decision tree (max depth = 5)
	model_tree5 = fit_model(df_train, model='tree', max_depth=5)
	accuracy_unseen_tree5, recall_unseen_tree5, precision_unseen_tree5, \
		cm_unseen_tree5, ninety_perc_tree5 = \
			evaluate_model_on_full_set(df_test, model_tree5)
	print '### DECISION TREE (max_depth=5) ###'
	print accuracy_unseen_tree5*100
	print recall_unseen_tree5*100
	print precision_unseen_tree5*100
	print ninety_perc_tree5
	print '-'*100

	# lda
	model_lda = fit_model(df_train, model='lda')
	accuracy_unseen_lda, recall_unseen_lda, precision_unseen_lda, \
		cm_unseen_lda, ninety_perc_lda = \
			evaluate_model_on_full_set(df_test, model_lda)
	print '### LDA ###'
	print accuracy_unseen_lda*100
	print recall_unseen_lda*100
	print precision_unseen_lda*100
	print ninety_perc_lda
	print '-'*100

	# nb
	model_nb = fit_model(df_train, model='naivebayes')
	accuracy_unseen_nb, recall_unseen_nb, precision_unseen_nb, \
		cm_unseen_nb, ninety_perc_nb = \
			evaluate_model_on_full_set(df_test, model_nb)
	print '### NAIVE BAYES ###'
	print accuracy_unseen_nb*100
	print recall_unseen_nb*100
	print precision_unseen_nb*100
	print ninety_perc_nb
	print '-'*100

	return accuracy_unseen_logreg, recall_unseen_logreg, precision_unseen_logreg, \
		accuracy_unseen_tree2, recall_unseen_tree2, precision_unseen_tree2, \
		accuracy_unseen_tree5, recall_unseen_tree5, precision_unseen_tree5, \
		accuracy_unseen_lda, recall_unseen_lda, precision_unseen_lda, \
		accuracy_unseen_nb, recall_unseen_nb, precision_unseen_nb, \
		ninety_perc_logreg, ninety_perc_tree2, ninety_perc_tree5, \
		ninety_perc_lda, ninety_perc_nb


def evaluate_flat_bid():
	print '### SIMULATING FLAT BID ###'
	print 'ad_spend, imp, click, cpm, ctr, cpc'
	ad_spend_vector = []
	imp_vector = []
	click_vector = []
	flat_bids = [5,10,20,30,40,50,75,100,150,200,250,300]
	for flat_bid in flat_bids:
		ad_spend, imp, click, cpm, ctr, cpc = \
			simulate_flat_bid(flat_bid, print_output=False)
		ad_spend_vector.append(ad_spend)
		imp_vector.append(imp)
		click_vector.append(click)
		print ad_spend, imp, click, cpm, ctr, cpc
	print
	print '-'*100

	return flat_bids, ad_spend_vector, imp_vector, click_vector


def evaluate_constant_goal_bid():
	print '### SIMULATING GOAL BID ###'

	# load the data
	df_train = pd.DataFrame.from_csv(DATA_PATH + 'train_sample_100.csv')

	# train the model
	model_logreg = \
		fit_model(df_train, model='logistic_regression')

	# simulate for multiple goals
	ad_spend_vector = []
	impressions_vector = []
	clicks_vector = []
	goals_vector = [10,25,50,75,100,150,200,500,1000]
	for g in goals_vector:
		print 'Goal: %s' % str(g)
		ad_spend, impressions, clicks, cpm, ctr, cpc = \
			simulate_goal_bid(goal=g,model=model_logreg)
		ad_spend_vector.append(ad_spend)
		impressions_vector.append(impressions)
		clicks_vector.append(clicks)
		print 'g, ad_spend, imp, click, cpm, ctr, cpc'
		print g, ad_spend, impressions, clicks, cpm, ctr,cpc

	return goals_vector, ad_spend_vector, impressions_vector, clicks_vector


def evaluate_min_cpe_bid(flat_goal_budgets):
	print '### SIMULATING DYNAMIC BID ###'

	# load the data
	df_train = pd.DataFrame.from_csv(DATA_PATH + 'train_sample_100.csv')

	# fit the model
	model_logreg = \
		fit_model(df_train, model='logistic_regression')

	# simulate for different budgets
	ad_spend_vector = []
	impressions_vector = []
	clicks_vector = []
	budgets_vector = flat_goal_budgets
	for b in budgets_vector:
		ad_spend, impressions, clicks, cpm, ctr, cpc = \
			simulate_dynamic_bid(budget=b, model=model_logreg)
		ad_spend_vector.append(ad_spend)
		impressions_vector.append(impressions)
		clicks_vector.append(clicks)

	return flat_goal_budgets, ad_spend_vector, impressions_vector, clicks_vector


def preprocess_data():
	df_imp_train_featureselection = feature_selection_imps(IMP_TRAIN_PATH)
	df_imp_train_featureselection.to_csv(DATA_PATH + 'train_imp_fs.csv')
	print 'train imps feature selection complete.'

	df_click_train_featureselection = feature_selection_clicks(CLICK_TRAIN_PATH)
	df_click_train_featureselection.to_csv(DATA_PATH + 'train_click_fs.csv')
	print 'train clicks feature selection complete.'

	df_train_merge = \
		merge_impression_click(df_imp_train_featureselection, df_click_train_featureselection)
	df_train_merge.to_csv(DATA_PATH + 'train_merge.csv')
	print 'train imp click merge complete.'

	df_train_ssf = transform_ssf(df_train_merge)
	df_train_ssf.to_csv(DATA_PATH + 'train_ssf.csv')
	print 'train ssf transform complete.'

	df_imp_test_featureselection = feature_selection_imps(IMP_TEST_PATH)
	df_imp_test_featureselection.to_csv(DATA_PATH + 'test_imp_fs.csv')
	print 'test imps feature selection complete.'

	df_click_test_featureselection = feature_selection_clicks(CLICK_TEST_PATH)
	df_click_test_featureselection.to_csv(DATA_PATH + 'test_click_fs.csv')
	print 'test clicks feature selection complete.'

	df_test_merge = merge_impression_click(df_imp_test_featureselection, df_click_test_featureselection)
	df_test_merge.to_csv(DATA_PATH + 'test_merge.csv')
	print 'test imp click merge complete.'

	df_test_ssf = transform_ssf(df_test_merge)
	df_test_ssf.to_csv(DATA_PATH + 'test_ssf.csv')
	print 'test ssf transform complete.'

	for ss_ratio in [100,200,500,1000,2000,5000]:
		print ss_ratio
		df_sample = sample_train_data(df_train_ssf, perc_no_clicks=ss_ratio)
		df_test = df_test_ssf.copy()

		# rename hour attributes in df_sample
		for c in df_sample.columns:
			if c[0:5] == 'hour_' and len(c) == 6:
				new_name = c[0:5] + '0' + c[-1]
				df_sample = df_sample.rename(columns={c: new_name})

		# sort the columns alphabetically
		df_test = df_test.reindex_axis(sorted(df_test.columns), axis=1)
		df_sample = df_sample.reindex_axis(sorted(df_sample.columns), axis=1)

		# remove train attributes if they do not show up in the test data
		for c in df_sample.columns:
			if c not in df_test.columns:
				df_sample = df_sample.drop([c], axis=1)

		# remove test attributes if they do not show up in the train data
		for c in df_test.columns:
			if c not in df_sample.columns:
				df_test = df_test.drop([c], axis=1)

		# write the new sets to csv
		df_test.to_csv(DATA_PATH + 'test_ssf_%s.csv' % str(ss_ratio))
		df_sample.to_csv(DATA_PATH + 'train_sample_%s.csv' % str(ss_ratio))



def predicting_propensity_to_click():
	accuracies_logreg = []
	accuracies_tree2 = []
	accuracies_tree5 = []
	accuracies_lda = []
	accuracies_nb = []
	recalls_logreg = []
	recalls_tree2 = []
	recalls_tree5 = []
	recalls_lda = []
	recalls_nb = []
	precisions_logreg = []
	precisions_tree2 = []
	precisions_tree5 = []
	precisions_lda = []
	precisions_nb = []
	ninety_cums_logreg = []
	ninety_cums_tree2 = []
	ninety_cums_tree5 = []
	ninety_cums_lda = []
	ninety_cums_nb = []


	ss_ratios = [100, 200, 500, 1000, 5000]
	# ss_ratios = [5000, 10000, 20000]
	for ssr in ss_ratios:
		df_train = pd.DataFrame.from_csv(DATA_PATH + 'train_sample_%s.csv' % str(ssr))
		df_test = pd.DataFrame.from_csv(DATA_PATH + 'test_ssf_%s.csv' % str(ssr))
		df_train = df_train.replace([np.inf, -np.inf], np.nan)
		df_train = df_train.dropna()
		df_test = df_test.replace([np.inf, -np.inf], np.nan)
		df_test = df_test.dropna()
		accuracy_unseen_logreg, recall_unseen_logreg, precision_unseen_logreg, \
			accuracy_unseen_tree2, recall_unseen_tree2, precision_unseen_tree2, \
			accuracy_unseen_tree5, recall_unseen_tree5, precision_unseen_tree5, \
			accuracy_unseen_lda, recall_unseen_lda, precision_unseen_lda, \
			accuracy_unseen_nb, recall_unseen_nb, precision_unseen_nb, \
			ninety_cum_logreg, ninety_cum_tree2, ninety_cum_tree5, \
			ninety_cum_lda, ninety_cum_nb = evaluate_models(df_train, df_test)
		accuracies_logreg.append(accuracy_unseen_logreg)
		accuracies_tree2.append(accuracy_unseen_tree2)
		accuracies_tree5.append(accuracy_unseen_tree5)
		accuracies_lda.append(accuracy_unseen_lda)
		accuracies_nb.append(accuracy_unseen_nb)
		recalls_logreg.append(recall_unseen_logreg)
		recalls_tree2.append(recall_unseen_tree2)
		recalls_tree5.append(recall_unseen_tree5)
		recalls_lda.append(recall_unseen_lda)
		recalls_nb.append(recall_unseen_nb)
		precisions_logreg.append(precision_unseen_logreg)
		precisions_tree2.append(precision_unseen_tree2)
		precisions_tree5.append(precision_unseen_tree5)
		precisions_lda.append(precision_unseen_lda)
		precisions_nb.append(precision_unseen_nb)
		ninety_cums_logreg.append(ninety_cum_logreg)
		ninety_cums_tree2.append(ninety_cum_tree2)
		ninety_cums_tree5.append(ninety_cum_tree5)
		ninety_cums_lda.append(ninety_cum_lda)
		ninety_cums_nb.append(ninety_cum_nb)

		print ssr

	# create charts
	plt.plot(ss_ratios, accuracies_logreg)
	plt.plot(ss_ratios, accuracies_tree2)
	plt.plot(ss_ratios, accuracies_tree5)
	plt.plot(ss_ratios, accuracies_lda)
	plt.plot(ss_ratios, accuracies_nb)
	plt.title('Test Accuracy At Different Strafied Sampling Thresholds')
	plt.ylabel('Accuracy')
	plt.xlabel('Strafied Sample %')
	plt.legend(['logreg', 'tree2', 'tree5', 'lda', 'nb'])
	plt.savefig(CHARTS_DIR_PATH + 'ss_accuracy.png')
	plt.clf()

	plt.plot(ss_ratios, recalls_logreg)
	plt.plot(ss_ratios, recalls_tree2)
	plt.plot(ss_ratios, recalls_tree5)
	plt.plot(ss_ratios, recalls_lda)
	plt.plot(ss_ratios, recalls_nb)
	plt.title('Test Recall At Different Strafied Sampling Thresholds')
	plt.ylabel('Recall')
	plt.xlabel('Strafied Sample %')
	plt.legend(['logreg', 'tree2', 'tree5', 'lda', 'nb'])
	plt.savefig(CHARTS_DIR_PATH + 'ss_recall.png')
	plt.clf()

	plt.plot(ss_ratios, precisions_logreg)
	plt.plot(ss_ratios, precisions_tree2)
	plt.plot(ss_ratios, precisions_tree5)
	plt.plot(ss_ratios, precisions_lda)
	plt.plot(ss_ratios, precisions_nb)
	plt.title('Test Precision At Different Strafied Sampling Thresholds')
	plt.ylabel('Precision')
	plt.xlabel('Strafied Sample %')
	plt.legend(['logreg', 'tree2', 'tree5', 'lda', 'nb'])
	plt.savefig(CHARTS_DIR_PATH + 'ss_precision.png')
	plt.clf()

	plt.plot(ss_ratios, ninety_cums_logreg)
	plt.plot(ss_ratios, ninety_cums_tree2)
	plt.plot(ss_ratios, ninety_cums_tree5)
	plt.plot(ss_ratios, ninety_cums_lda)
	plt.plot(ss_ratios, ninety_cums_nb)
	plt.title('Ninety Percent Cumulative Click At Different Strafied Sampling Thresholds')
	plt.ylabel('90%% Cumulative Click')
	plt.xlabel('Strafied Sample %')
	plt.legend(['logreg', 'tree2', 'tree5', 'lda', 'nb'])
	plt.savefig(CHARTS_DIR_PATH + 'ss_ninety.png')
	plt.clf()


	# print type(ss_ratios), type(fscores_logreg)
	# plt.plot(ss_ratios, fscores_logreg)
	# plt.plot(ss_ratios, fscores_tree2)
	# plt.plot(ss_ratios, fscores_tree5)
	# plt.plot(ss_ratios, fscores_lda)
	# plt.plot(ss_ratios, fscores_nb)
	# plt.title('F Score At Different Strafied Sampling Thresholds')
	# plt.ylabel('F Score')
	# plt.xlabel('Strafied Sample %')
	# plt.legend(['logreg', 'tree2', 'tree5', 'lda', 'nb'])
	# plt.savefig(CHARTS_DIR_PATH + 'ss_fscore.png')
	# plt.clf()


preprocess_data()
predicting_propensity_to_click()
flat_bids, ad_spend_fb, imps_fb, clicks_fb = evaluate_flat_bid()
goals, ad_spend_fg, imps_fg, clicks_fg = evaluate_constant_goal_bid()
budgets, ad_spend_dg, imps_dg, clicks_dg = evaluate_min_cpe_bid(ad_spend_fg)

df_fb = pd.DataFrame(columns=['flat_bid', 'ad_spend', 'imps', 'clicks'])
df_fb['flat_bid'] = flat_bids
df_fb['ad_spend'] = ad_spend_fb
df_fb['imps'] = imps_fb
df_fb['clicks'] = clicks_fb
df_fb['cpm'] = df_fb['ad_spend'].divide(df_fb['imps'])*1000
df_fb['ctr'] = df_fb['clicks'].divide(df_fb['imps'])*100
df_fb['cpc'] = df_fb['ad_spend'].divide(df_fb['clicks'])

df_fg = pd.DataFrame(columns=['goal', 'ad_spend', 'imps', 'clicks'])
df_fg['goal'] = goals
df_fg['ad_spend'] = ad_spend_fg
df_fg['imps'] = imps_fg
df_fg['clicks'] = clicks_fg
df_fg['cpm'] = df_fg['ad_spend'].divide(df_fg['imps'])*1000
df_fg['ctr'] = df_fg['clicks'].divide(df_fg['imps'])*100
df_fg['cpc'] = df_fg['ad_spend'].divide(df_fg['clicks'])

df_dg = pd.DataFrame(columns=['budget', 'ad_spend', 'imps', 'clicks'])
df_dg['budget'] = budgets
df_dg['ad_spend'] = ad_spend_dg
df_dg['imps'] = imps_dg
df_dg['clicks'] = clicks_dg
df_dg['cpm'] = df_dg['ad_spend'].divide(df_dg['imps'])*1000
df_dg['ctr'] = df_dg['clicks'].divide(df_dg['imps'])*100
df_dg['cpc'] = df_dg['ad_spend'].divide(df_dg['clicks'])

# flat bid
fig, ax1 = plt.subplots()
ax1.plot(df_fb['flat_bid'], df_fb['cpc'], marker='o')
ax1.plot(0,0,color='#2ca25f', marker='^')
ax1.set_xlabel('Flat Bid (Yen)')
ax1.set_ylabel('Cost Per Click (Yen)')
ax2 = ax1.twinx()
ax2.plot(df_fb['flat_bid'], df_fb['ctr'], color='#2ca25f', marker='^')
ax2.set_ylabel('Click Through Rate (%)')
ax1.legend(['cpc','ctr'])
plt.title('Performance At Varying Flat Bid Amounts')
plt.savefig(CHARTS_DIR_PATH + 'flatbid.png')
plt.clf()

# flat goal
fig, ax1 = plt.subplots()
ax1.plot(df_fg['goal'], df_fg['cpc'], marker='o')
ax1.plot(0,0,color='#2ca25f', marker='^')
ax1.set_xlabel('Goal (Yen)')
ax1.set_ylabel('Cost Per Click (Yen)')
ax2 = ax1.twinx()
ax2.plot(df_fg['goal'], df_fg['ctr'], color='#2ca25f', marker='^')
ax2.set_ylabel('Click Through Rate (%)')
ax1.legend(['cpc','ctr'])
plt.title('Performance At Varying Flat Goal Amounts')
plt.savefig(CHARTS_DIR_PATH + 'flatgoal.png')
plt.clf()

# dynamic goal
fig, ax1 = plt.subplots()
ax1.plot(df_dg['budget'], df_dg['cpc'], marker='o')
ax1.plot(0,0,color='#2ca25f', marker='^')
ax1.set_xlabel('Budget (Yen)')
ax1.set_ylabel('Cost Per Click (Yen)')
ax2 = ax1.twinx()
ax2.plot(df_dg['budget'], df_dg['ctr'], color='#2ca25f', marker='^')
ax2.set_ylabel('Click Through Rate (%)')
ax1.legend(['cpc','ctr'])
plt.title('Performance At Varying Budgets')
plt.savefig(CHARTS_DIR_PATH + 'dynamicgoal.png')
plt.clf()

# plot cpm vs ad spend
plt.plot(df_fb['ad_spend'], df_fb['cpm'], marker='o')
plt.plot(df_fg['ad_spend'], df_fg['cpm'], marker='^')
plt.plot(df_dg['ad_spend'], df_dg['cpm'], marker='s')
plt.title('CPM At Varying Budget Levels')
plt.xlabel('Ad Spend (Yen)')
plt.ylabel('Cost Per Thousand Impressions (Yen)')
plt.legend(['flat bid', 'flat goal', 'dynamic goal'])
plt.savefig(CHARTS_DIR_PATH + 'cpm_vs_adspend.png')
plt.clf()

# plot ctr vs ad spend
plt.plot(df_fb['ad_spend'], df_fb['ctr'], marker='o')
plt.plot(df_fg['ad_spend'], df_fg['ctr'], marker='^')
plt.plot(df_dg['ad_spend'], df_dg['ctr'], marker='s')
plt.title('Click Through Rate At Varying Budget Levels')
plt.xlabel('Ad Spend (Yen)')
plt.ylabel('Click Through Rate (%)')
plt.legend(['flat bid', 'flat goal', 'dynamic goal'])
plt.savefig(CHARTS_DIR_PATH + 'ctr_vs_adspend.png')
plt.clf()

# plot cpc vs ad spend
plt.plot(df_fb['ad_spend'], df_fb['cpc'], marker='o')
plt.plot(df_fg['ad_spend'], df_fg['cpc'], marker='^')
plt.plot(df_dg['ad_spend'], df_dg['cpc'], marker='s')
plt.title('Cost Per Click At Varying Budget Levels')
plt.xlabel('Ad Spend (Yen)')
plt.ylabel('Cost Per Click (Yen)')
plt.legend(['flat bid', 'flat goal', 'dynamic goal'])
plt.savefig(CHARTS_DIR_PATH + 'cpc_vs_adspend.png')
plt.clf()

