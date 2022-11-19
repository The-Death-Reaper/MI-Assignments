'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	if(len(df)==0):
		return entropy
	target_col = df[df.columns[-1]]
	d = {}
	total = 0
	for i in target_col:
		total += 1
		try:
			d[i] = d[i] + 1
		except:
			d[i] = 1
	for i in d.values():
		entropy += -((i/total)*np.log2(i/total))
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large

def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	if(len(df)==0):
		return entropy_of_attribute
	att_column = df[attribute]
	d = {}
	total = 0
	for i in att_column:
		total += 1
		try:
			d[i] = d[i] + 1
		except:
			d[i] = 1
	for k,v in d.items():
		entropy_of_attribute += -(v/total)*get_entropy_of_dataset(df[df[attribute]==k])
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
	
def get_information_gain(df,attribute):
	entropy_target = get_entropy_of_dataset(df)
	entropy_attribute = get_entropy_of_attribute(df, attribute)
	information_gain = entropy_target - entropy_attribute
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')
    
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''
	max_info_gain = -1
	if(len(df)==0):
		return (information_gains,selected_column)
	for i in df.columns[:len(df.columns)-1:]:
		information_gains[i] = get_information_gain(df, i)
		if information_gains[i] > max_info_gain:
			max_info_gain = information_gains[i]
			selected_column = i

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
