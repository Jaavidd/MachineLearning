import numpy as np
import pandas as pd
import statsmodels
import sklearn.cluster as KMeans
import  matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set()
pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',200)
import math
import pprint
dt=pd.read_csv("iris.csv")

# setosa=pd.read_csv("iris_text.data",nrows=49)
# data=pd.DataFrame(data,columns=['petal_w' , 'petal_l', 'sepal_w' , 'sepal_l','class'])

# print(setosa)
#
# versicolor=pd.read_csv("iris_text.data",skiprows=50,nrows=49)
#
# virginica=pd.read_csv("iris_text.data",skiprows=100)
#
#
#
#
# petal_w=data.iloc[:,0]
# petal_l=data.iloc[:,1]
# sepal_w=data.iloc[:,2]
# sepal_l=data.iloc[:,3]
#
dt=dt.rename(columns={"species":"label"})

def train_test(dt,test_size):
    if isinstance(test_size,float):
        test_size=round(test_size*len(data))

    indices= dt.index.tolist()
    test_ind=random.sample(population=indices,k=test_size)

    test_df=dt.loc[test_ind]
    train_df=dt.drop(test_ind)


    return train_df,test_df

random.seed(0)
train_df,test_df = train_test(dt,test_size=20)


data=train_df.values

def check_purity(data):
    label_col = data[:, -1]
    unique_classes = np.unique(label_col)
    if len(unique_classes)==1:
        return True

    return False


def classify_data(data):
    label_col = data[:, -1]
    classif=label_col[0]
    unique_classes,count=np.unique(label_col,return_counts=True)

    ind=count.argmax()
    classif=unique_classes[ind]

    return classif

# print(classify_data(train_df[train_df['petal.width']>0.8].values))

def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for col_ind in range(n_columns - 1):
        potential_splits[col_ind] = []
        values = data[:, col_ind]

        unique_val = np.unique(values)

        for index in range(len(unique_val)):
            if index != 0:
                current_val = unique_val[index]
                previous_val = unique_val[index - 1]
                pot_split = (current_val + previous_val) / 2

                potential_splits[col_ind].append(pot_split)


    return potential_splits

potential_splits=get_potential_splits(train_df.values)
sns.lmplot(data=train_df,x="petal.width",y="petal.length",hue='label',fit_reg=False,height=6,aspect=1.5)
# plt.vlines(x=potential_splits[3],ymin=1,ymax=7)
# plt.hlines(y=potential_splits[2],xmin=0,xmax=2.5)
# plt.show()
plt.close()

def split_data(data,split_columns,split_val):
    split_col_values = data[:, split_columns]

    data_below = data[split_col_values <= split_val]
    data_above = data[split_col_values > split_val]
    # pass
    return data_below,data_above

split_col=3
split_val=0.8

data_below,data_above=split_data(data,split_col,split_val)
plotting_df=pd.DataFrame(data,columns=dt.columns)
sns.lmplot(data=plotting_df,x='petal.width',y='petal.length',fit_reg=False,height=6,aspect=1.5)
plt.vlines(x=split_val,ymin=1,ymax=7)
# plt.show()

"""""ENTROPY"""

def calculate_entropy(data):
    label_col = data[:, -1]
    _, counts = np.unique(label_col, return_counts=True)

    probabilities = counts / sum(counts)

    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

Entropy=calculate_entropy(data)

def overall_entropy(data_below,data_above):
    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points

    overall_ent = (p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))


    return overall_ent


# print(overall_entropy(data_below,data_above))

def find_BestSplit(data,potential_splits):
    overallE = 999
    # split_col = 0
    # split_value =0
    for col in potential_splits:
        for val in potential_splits[col]:
            data_below, data_above = split_data(data, split_columns=col, split_val=val)
            current = overall_entropy(data_below, data_above)

            if current <= overallE:
                overallE = current
                split_col=col
                split_value=val


    return split_col,split_value



def EntropyGroup(data):
    nSpecies=[0,0,0]
    H_g=0
    n_setosa=0
    n_versic=0
    n_virgin=0
    for item in data["label"]:

        if item=="Virginica":
            nSpecies[0]+=1
            n_virgin+=1
        elif item=='Setosa':
            nSpecies[1]+=1
            n_setosa+=1
        elif item=='Versicolor':
            nSpecies[2]+=1
            n_versic+=1
        else:
            print("BUUUUUUUUUUUUUUG")

    for i in range(0,3):
        tmp=nSpecies[i]/len(data)
        if tmp != 0:
            H_g-=tmp*math.log(tmp,2)

    return H_g,[n_setosa,n_versic,n_virgin]

# print(dt)

def discriminativePower(data,H,attr,G):
    # disc=0.0+H

    partitions=len(data)//G
    sor_data=data.sort_values(by=[attr])

    g1=sor_data[:partitions]
    g2=sor_data[partitions:2*partitions]
    g3=sor_data[partitions*2 : 3* partitions+1]

    H_g=[0,0,0]
    number=[0,0,0]
    H_g[0],number[0]=EntropyGroup(g1)
    H_g[1] ,number[1]= EntropyGroup(g2)
    H_g[2] ,number[2]= EntropyGroup(g3)



    # print(g1)
    return list(H_g),number

entropy,avoid=EntropyGroup(dt)

disc_power,number=discriminativePower(dt,entropy,"sepal.length",3)
# print(number)
disc_p=entropy- 50/150*sum(disc_power)
# print(disc_power)
def print_result():
    print()

    for i in ['sepal.length','sepal.width','petal.length','petal.width']:

        disc_power,number=discriminativePower(dt,entropy,i,3)
        # print(disc_power)
        disc_p=entropy-float(50/150)*sum(disc_power)

        print("Data entropy: ",entropy," Disc= ",disc_p)
        for j,k in zip(number,disc_power):
            print("[50]","Setosa:",j[0],"Versicolor: ",j[1]," Virginica: ",j[2]," Entropy=",k)

        print()
print_result()




# DECISION TREE ALGORITHM


"""Map for Tree"""
example_tree={"petal.width <=0.8":["Setosa",{"petal.width <=1.65":
        [{"petal.length <=4.9":["Versicolor","Virginica"]},
                                "Virginica"]}]}

# Recursive solution
def decision_tree(df,counter=0,min_sample=2,max_depth=5):

    if counter==0:
        global column_names
        column_names=df.columns
        data=df.values

    else:
        data=df

    if (check_purity(data)) or (len(data)<min_sample) or (counter==max_depth):
        # classification=classify_data(data)
        return classify_data(data)

    else:
        counter += 1

        pot_splits=get_potential_splits(data)
        split_column,split_value=find_BestSplit(data,pot_splits)
        data_below,data_above=split_data(data,split_column,split_value)


        # first create sub-tree and clarify questions for decisions
        feature_name=column_names[split_column]

        question="{} <= {}".format(feature_name,split_value)
        sub_tree={question: []}

        # answers yes || no

        yes_answer=decision_tree(data_below,counter,min_sample,max_depth)
        no_answer=decision_tree(data_above,counter,min_sample,max_depth)

        if yes_answer == no_answer:
            sub_tree=yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree



tree=decision_tree( train_df,max_depth=3)

# pprint.pprint(tree)

# classification
example=test_df.iloc[2]
# print(example)
def classify_example(example,tree):
    question = list((tree.keys()))[0]
    feature_name, comparison, value = question.split()

    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    else:
        residual_tree = answer
        return classify_example(example, residual_tree)



# print(classify_example(example,tree))

# accuracy
def calculate_accuracy(dt,tree):
    dt["classification"] = dt.apply(classify_example,axis=1,args=(tree,))
    dt["classification_correct"]=dt.classification==dt.label
    accuracy=dt.classification_correct.mean()

    return accuracy

# print(calculate_accuracy(test_df,tree))
# print(test_df.loc[77])