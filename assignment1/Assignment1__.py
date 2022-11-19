'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

def calc_entropy(count_yes,count_no):
	tot_samples = count_yes + count_no
	pos_ratio = count_yes/tot_samples
	neg_ratio = count_no/tot_samples
	
	if pos_ratio!=0 and neg_ratio!=0:
		entropy = -((pos_ratio)*(np.log2(pos_ratio))) - ((neg_ratio)*(np.log2(neg_ratio)))
	elif pos_ratio==0:
		entropy = - ((neg_ratio)*(np.log2(neg_ratio)))
	elif neg_ratio==0:
		entropy = -((pos_ratio)*(np.log2(pos_ratio)))
	return entropy
		
'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	df.dropna()
	num_col = len(df.columns)
	df2 = df[df.columns[num_col - 1]]
	count_yes = 0
	count_no = 0
	for i in df2:
		if i.lower()=='yes':
			count_yes += 1
		elif i.lower()=='no':
			count_no += 1

	entropy = calc_entropy(count_yes,count_no)
	# tot_samples = count_yes + count_no
	# pos_ratio = count_yes/tot_samples
	# neg_ratio = count_no/tot_samples

	# entropy = -(pos_ratio)*(np.log2(pos_ratio)) - (neg_ratio)*(np.log2(neg_ratio))

	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):

	entropy_of_attribute = 0
	df.dropna()
	num_col = len(df.columns)
	df2 = df[[attribute,df.columns[num_col - 1]]]
	arr = df2.values
	set_attr = set()
	for i in arr:
		set_attr.add(i[0])
	d = {}
	for j in set_attr:
		d[j] = [0,0]
	
	for k in arr:
		if k[1].lower()=='yes':
			d[k[0]][0]+=1
		elif k[1].lower()=='no':
			d[k[0]][1]+=1
	#print(d)
	tot_samples = len(arr)
	for c in d:
		mul = (d[c][0]+d[c][1])/tot_samples
		entropy_of_attribute += (calc_entropy(d[c][0],d[c][1]))*mul

	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0
	entropy_dataset = get_entropy_of_dataset(df)
	entropy_attr = get_entropy_of_attribute(df,attribute)
	information_gain = entropy_dataset - entropy_attr
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):

	information_gains={}
	selected_column=''
	max_inf_gain = -1
	i = 0
	num_col = len(df.columns)

	for col in df.columns:
		if i!=num_col-1:
			inf_gain = get_information_gain(df,col)
			information_gains[col] = inf_gain
			if inf_gain>max_inf_gain:
				max_inf_gain = inf_gain
				selected_column = col
		i+=1

	#print(information_gains,selected_column)
	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''