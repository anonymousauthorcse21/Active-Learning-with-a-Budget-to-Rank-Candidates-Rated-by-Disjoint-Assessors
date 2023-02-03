from util import *
from Algorithms import *
from plots import *

def getDataSet(datasetName):
    """
    It takes a string as input and returns a numpy array
    
    :param datasetName: The name of the dataset to be used
    :return: The dataset is being returned.
    """
    if datasetName == "sd1":
        dataset = np.loadtxt('Datasets\Synthetic_DataSet_1(Gaussian_Weightage).csv', delimiter =',')
    
    elif datasetName == "sd2":
        dataset = np.loadtxt('Datasets\Synthetic_DataSet_1(Gaussian_Weightage).csv', delimiter =',')
    elif datasetName == "MIT":
        dataset = np.loadtxt('Datasets\MIT_Dataset.csv', delimiter =',')
    return dataset


if __name__=='__main__':
    datasetName = input("Which DataSet to Test?\n \
        1. For Synthetic Dataset 1 : Enter -> sd1 \n \
        2. For Synthetic Dataset 2 : Enter -> sd2 \n \
        3. For MIT Dataset : Enter -> MIT \n")

    Rstar = getDataSet(datasetName)
    No = Rstar.shape[0]
    Na = Rstar.shape[1]
    Quality_Metric_values = []
    nDCG_Metric_values = []
    
    
    quality_at_k,ndcg_at_k = Blk_SA(Rstar,datasetName)  # BASELINE 1
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_rand_Bgt(Rstar,No,Na,datasetName) # BASELINE 2
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_Lc_Bgt(Rstar,No,Na,datasetName,rank=3) # BASELINE 3
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_rand_Lc_Bgt(Rstar,datasetName,No,Na,rank=3) # BASELINE 4
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_comp(Rstar,datasetName) # BASELINE 5
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_rand_Bgt_comp(Rstar,datasetName,No,Na) # BASELINE 6
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_Lc_Bgt_comp(Rstar,No,Na,datasetName,rank=3) # BASELINE 7
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = Blk_rand_Lc_Bgt_comp(Rstar,No,Na,datasetName,rank=3,extraBudget=100) # BASELINE 8
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = OPLP_Query(Rstar,No,Na,datasetName,rank=3) #OPLP-Query
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)

    quality_at_k,ndcg_at_k = BFLP_Query(Rstar,No,Na,datasetName,rank=3,extraBudget=100)  #BFLP-Query
    Quality_Metric_values.append(quality_at_k)
    nDCG_Metric_values.append(ndcg_at_k)



    # allAlgoplot(nDCG_Metric_values,"darkgreen",1,"AllAlgo_vs_nDCG25__"+datasetName)   #Plot nDCG@25 vs Algorithms
    # trPath = "tr_g.csv"     
    # prPath = 'pr_g_ALGORITHM_NAME.csv'     # Enter Algorithm name from ['block' , 'OPLP_Query' ,'BFLP_Query']
    # plot_tr_vs_pr(trPath,prPath,No=200)    #No=138 for MIT DATA, No = 200 for synthetic data


                

