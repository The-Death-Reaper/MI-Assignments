from Assignment1 import *   

def test_case():
    toothed='true,true,true,false,true,true,true,true,true,false'.split(',')
    breathes='true,true,true,true,true,true,false,true,true,true'.split(',')
    legs='true,true,false,true,true,true,false,false,true,true'.split(',')
    species='mammal,mammal,reptile,mammal,mammal,mammal,reptile,reptile,mammal,reptile'.split(',')

    dataset ={'toothed':toothed,'breathes':breathes,'legs':legs,'species':species}
    df = pd.DataFrame(dataset,columns=['toothed','breathes','legs','species'])
    print(get_entropy_of_dataset(df))
    print(get_entropy_of_attribute(df,'toothed'))
    print(get_entropy_of_attribute(df,'breathes'))
    print(get_entropy_of_attribute(df,'legs'))
    print(get_information_gain(df,'toothed'))
    print(get_information_gain(df,'breathes'))
    print(get_information_gain(df,'legs'))

if __name__=="__main__":
	test_case()