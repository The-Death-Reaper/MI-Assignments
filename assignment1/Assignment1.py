'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

def shannon_entropy_series(counts):
	t_sum = counts.sum()
	counts = counts/t_sum
	entropy = (counts*np.log2(counts))
	return -1 * entropy.sum()


'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	temp = df.iloc[:,-1]
	counts = temp.value_counts()
	entropy = shannon_entropy_series(counts)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	col = df[attribute]
	counts = col.value_counts()
	t_sum = counts.sum()
	for k,v in counts.items():
		entropy_of_attribute += (v/t_sum)*get_entropy_of_dataset(df[df[attribute] == k])
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df,attribute)
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	cols = df.columns
	cols = cols[:len(cols) - 1:]
	max_info_gain = 0
	selected_column = ''
	for col in cols:
		temp = get_information_gain(df,col)
		information_gains[col] = temp
		if temp > max_info_gain:
			max_info_gain = temp
			selected_column = col

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