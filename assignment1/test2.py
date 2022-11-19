from Assignment1_ import *   

def test_case():
    # salary = 'tier1,tier2,tier1,tier1,tier2,tier1,tier1'.split(',')
    salary = ['tier1', 'tier1']
    # location = 'mum,blr,blr,hyd,mum,hyd,hyd'.split(',')
    location = ['mum', 'blr']
    # job = 'yes,yes,no,no,yes,no,no'.split(',')
    job = ['yes','yes']
    dataset ={'salary':salary,'location':location,'job':job}
    df = pd.DataFrame(dataset,columns=['salary','location','job'])

    print(get_entropy_of_dataset(df))
    print(get_entropy_of_attribute(df,'salary'))
    print(get_entropy_of_attribute(df,'location'))
    print(get_information_gain(df,'salary'))
    print(get_information_gain(df,'location'))
    print(get_selected_attribute(df))


if __name__=="__main__":
	test_case()