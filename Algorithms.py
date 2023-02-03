from util import *


def blockMask(dataset):
    """
    It creates a mask for the data, which is a list of lists, where each list is a list of 1's and 0's. 
    :param dataset: the name of the dataset you want to use
    :return: A numpy array of the mask.
    """
    mask=[]
    if dataset == "MIT":
        p1 = [1]*2 + [0]*7
        p2 = [0]*2 + [1]*3 + [0]*4
        p3 = [0]*5 + [1]*2 + [0]*2
        p4 = [0]*7 + [1]*2 
        mask = [p1]*34 + [p2]*34+ [p3]*35 + [p4]*35
        
    else:
        p1 = [1]*6 + [0]*24
        p2 = [0]*6 + [1]*6 + [0]*18
        p3 = [0]*6 +[0]*6 + [1]*6 + [0]*12
        p4 = [0]*6 + [0]*6 +[0]*6 + [1]*6 + [0]*6
        p5 = [0]*6 + [0]*6 + [0]*6 + [0]*6 + [1]*6
        mask = [p1]*40 + [p2]*40+ [p3]*40 + [p4]*40 + [p5]*40

    mask = np.array(mask)
    return mask

def blockMask_with_randomQuery(dataset,extraBudget,No,Na):
    """
    It takes a dataset, extra budget, number of objects and number of assessor as input and returns a
    mask with extra budget number of random queries
    
    :param dataset: the dataset to be masked
    :param extraBudget: the number of extra queries we want to make
    :param No: Number of objects
    :param Na: number of assessors
    :return: A mask of size No x Na
    """
    mask = blockMask(dataset)
    budget = extraBudget
    while(budget):
        i = np.random.randint(0,No-1)
        j = np.random.randint(0,Na-1)
        if(mask[i][j]==0): 
            mask[i][j] = 1
            budget-=1
    mask = np.array(mask)
    return mask

def saveRanks(pr,name,flag='pr'): 
    """
    To generate trueRank_vs_predictRank plots
    """
    pr = np.asarray(pr)
    if flag == 'pr':
        np.savetxt('pr_g_'+name+'.csv', pr, delimiter=",")
    else:
        np.savetxt('tr_g.csv', pr, delimiter=",")



def Calculate_Quality_at_k(trueRanking,predictedRanking,k=25):
    """
    It takes the true ranking and the predicted ranking and returns the percentage of the top k items in
    the predicted ranking that are also in the true ranking
    
    :param trueRanking: The true ranking of the items
    :param predictedRanking: The predicted ranking of the items
    :param k: The number of top-ranked items to consider for the quality metric, defaults to 25
    (optional)
    :return: The percentage of the top k items in the true ranking that are also in the top k items in
    the predicted ranking.
    """
    count = 0
    for st in trueRanking[:k]:
        if st in predictedRanking[:k]:
            count+=1
    return count/k * 100

def Calculate_dcg_at_k(result):
    """
    The function takes in a list of relevance scores and returns the discounted cumulative gain (DCG) at
    k.
    
    :param result: the list of relevance scores for the documents in the result list
    :return: The dcg scores 
    """
    dcg = []
    for idx, val in enumerate(result): 
        numerator = val
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)

def Calculate_ndcg_at_k(result, sorted_result): 
    """
    The function takes in two lists, the first list is the predicted relevance score, and the second list is
    the true relevance score. 
    The function returns the NDCG.
    
    :param result: predicted relevance score
    :param sorted_result: True relevance score
    :return: The ndcg is being returned.
    """
    dcg = Calculate_dcg_at_k(result)
    idcg = Calculate_dcg_at_k(sorted_result)
    ndcg = dcg / idcg
    return ndcg

def get_RelevanceScore_of_rankings(predictedRank,trueRanking,k=25):
    """
    It takes the predicted ranking and the true ranking and returns the true relevance and the predicted
    relevance. 
    
    :param predictedRank: The predicted ranking of the Objects
    :param trueRanking: The true ranking of the Objects 
    :param k: The number of top-ranked objects to consider for the evaluation, defaults to 25 (optional)
    :return: the true relevance and predicted relevance of the top k Objects.
    """
    trueRelevance = np.arange(1,k+1)
    trueRelevance = sorted(trueRelevance,reverse = True)
    predRelevance = []
    for i in predictedRank[:k]:
        if i in trueRanking[:k]:
            predRelevance.append(k - list(trueRanking[:k]).index(i))
        else:
            predRelevance.append(0)
    return trueRelevance,predRelevance


def getLocalCoherence(U,Vt,r,c0,No,Na):
    """
    > The function takes in the U, Vt, r, c0, No, and Na and returns the local coherence matrix
    
    :param U: The U matrix from the SVD
    :param Vt: The transpose of the matrix V
    :param r: rank of the matrix
    :param c0: a constant
    :param No: Number of objects
    :param Na: number of assessors
    :return: The local coherence probability matrix.
    """
    mu_i = [0]*No
    ei = np.identity(No)
    ej = np.identity(Na)

    for i in range(No):
        mu_i[i] = ((np.linalg.norm(U.T @ ei[i]))**2) * (No/r)

    vj = [0]*Na
    for i in range(Na):
        vj[i] = ((np.linalg.norm(Vt @ ej[i]))**2) * (Na/r)

    P_LC = []
    for l in range(No):
        for k in range(Na):
            f  = (c0 * (mu_i[l]+vj[k])*r*(math.log(No+Na))*(math.log(No+Na))) / Na
            P_LC.append(min(f,1))

    P_LC = np.array(P_LC)
    P_LC = P_LC.reshape(No,Na)
    return P_LC

def driver(r,Rstar,R,mask,c0,No,Na):
    """

    :param r: the rank of the matrix
    :param Rstar: The original matrix
    :param R: the masked matrix 
    :param mask: masks 
    :param c0: the constant
    :param No: number of Objects
    :param Na: number of Assessors
    :return: The local coherence of the matrix R.
    """
    import random
    R = np.array(R)
    u, s, vh = np.linalg.svd(R, full_matrices=True)
    U = u[:, :r] 
    s1 = np.diag(s[:r]) 
    Vt =  vh[:r, :]
    P_LC = getLocalCoherence(U,Vt,r,c0,No,Na)
    return P_LC

def coinFlip(p):      
    result = np.random.binomial(1,p) 
    return result

def nncComplete(Rstar_tensor,mask_tensor):
    """
    Returns the completed matrix using Nuclear Norm Minimization Method.
    
    :param Rstar_tensor: the tensor of the original matrix
    :param mask_tensor: a tensor of the same size as Rstar_tensor, with 1s where the values are known
    and 0s where the values are unknown
    :return: The original matrix and the completed matrix.
    """
    device = 'cuda'
    h, w = Rstar_tensor.shape
    r = min(h,w)
    L = nn.Parameter(torch.randn(h, r).to(device))
    Rt = nn.Parameter(torch.randn(r, w).to(device))

    optimizer = optim.Adam([L, Rt], lr=0.05)
    X = Rstar_tensor.to(device)

    gif_images = []
    interval = 50  #for gif
    reg_lamb = 100  #lambda/2

    for i in range(1000):
        Z = L @ Rt
        rec_loss = torch.norm((Z - X)*mask_tensor)**2
        reg_loss = reg_lamb * (torch.norm(L)**2 + torch.norm(Rt)**2)
        loss = rec_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if i % interval==0:
                gif_images.append(((Z.detach().cpu().numpy())))


    with torch.no_grad():
        Z = L @ Rt
        Z = Z.to('cpu')
    return X,Z


#BASELINES

def Blk_SA(Rstar,dataset):
    """
    Baseline Algorithm -1
    :param Rstar: The true ground data matrix
    :param dataset: The dataset to be used
    :return: Quality_at_k,nDCG_Value*100
    """
    mask = blockMask(dataset)
    R = Rstar * mask
    trueRank = np.argsort(sum(Rstar,axis = 1))[::-1]
    predRank = np.argsort(sum(R,axis = 1))[::-1]
    trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
    nDCG_Value = Calculate_ndcg_at_k(predRelevance, trueRelevance)
    Quality_at_k =  Calculate_Quality_at_k(trueRank,predRank)
    
    saveRanks(predRank,"block",'pr')
    saveRanks(trueRank,"block",'tr')

    print(f"Quality for Block = {Quality_at_k}")
    print(f"nDCG for Blk_SA = {nDCG_Value*100}")
    print("\n----------------------------------")
    return Quality_at_k,nDCG_Value*100

def Blk_rand_Bgt(Rstar,No,Na,dataset,extraBudget=100):
    """
    Baseline - 2
    
    :param Rstar: The true ground truth matrix
    :param No: Number of objects
    :param Na: Number of Assessors
    :param dataset: The dataset to be used
    :param extraBudget: The extra budget that we are going to use for the active query, defaults
    to 100 (optional)
    :return: The mean quality and mean nDCG for the Blk-rand-Bgt algorithm.
    """
    trueRank = np.argsort(sum(Rstar,axis = 1))[::-1]
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        mask = blockMask_with_randomQuery(dataset,extraBudget,No,Na)
        R = Rstar * mask

        Score = []
        for i in range(No):
            sumr = 0
            cnt =0
            for j in range(Na):
                sumr+=R[i][j]
                if(R[i][j] > 0):
                    cnt+=1

            Score.append(sumr/cnt)


        predRank = np.argsort(Score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))
        quality_each_epoch.append(quality_at_k)
        

    print(f"Quality for Blk-rand-Bgt= {np.mean(quality_each_epoch)}")
    print(f"nDCG for Blk-rand-Bgt =  {np.mean(nDCG_each_epoch)*100}")
    print("----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100



def Blk_Lc_Bgt(Rstar,No,Na,dataset,rank=3,extraBudget=100):
    """
    Baseline-3 
    
    :param Rstar: The true ground truth  matrix
    :param No: Number of objects
    :param Na: Number of assessors
    :param dataset: The dataset to be used
    :param rank: The rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The extra budget that we are going to use for active query, defaults to 100
    (optional)
    :return: Quality@25 and nDCG@25
    """
    mask = blockMask(dataset)
    R = Rstar * mask
    trueRank = np.argsort(sum(Rstar,axis = 1))[::-1]
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    c0 = 0.1
    print("\nEpoch#")
    for epoch in tqdm(range(100)):
        P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
        R_copy = R.copy()
        mask_copy = mask.copy()
        budget = extraBudget
        while(budget):
            i = np.random.randint(No)
            j = np.random.randint(Na)
            while(mask_copy[i][j] == 1):
                i = np.random.randint(No)
                j = np.random.randint(Na)
            probability = P_LC[i][j]
            shouldSample = coinFlip(probability)
            if(shouldSample):
                mask_copy[i][j] = 1
                budget-=1
        R_copy = Rstar * mask_copy
        Score = []
        for i in range(No):
            sumr = 0
            cnt =0
            for j in range(Na):
                sumr+=R_copy[i][j]
                if(R_copy[i][j] > 0):
                    cnt+=1

            Score.append(sumr/cnt)

        predRank = np.argsort(Score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))

    print(f"Quality for Blk_Lc_Bgt = {np.mean(quality_each_epoch)}")
    print(f"nDCG for Blk_Lc_Bgt = {np.mean(nDCG_each_epoch)*100}")
    print("----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100

def Blk_rand_Lc_Bgt(Rstar,dataset,No=200,Na=30,rank=3,extraBudget=100):
    """
    Baseline - 4
    
    :param Rstar: The true ground truth matrix
    :param dataset: The dataset to be used
    :param No: Number of objects, defaults to 200 (optional)
    :param Na: Number of assessors, defaults to 30 (optional)
    :param rank: The rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The extra budget that we are allowed to spend on Active Query, defaults to 100
    (optional)
    :return: Quality@25 and nDCG@25
    """
    trueRank = np.argsort(sum(Rstar,axis = 1))[::-1]
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    c0 = 0.1
    print("\nEpoch#")
    for epoch in tqdm(range(100)):
        mask = blockMask_with_randomQuery(dataset,extraBudget,No,Na)
        R = Rstar * mask
        P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
        budget = extraBudget/2
        while(budget):
            i = np.random.randint(No)
            j = np.random.randint(Na)
            while(mask[i][j] == 1):
                i = np.random.randint(No)
                j = np.random.randint(Na)
            probability = P_LC[i][j]
            shouldSample = coinFlip(probability)
            if(shouldSample):
                mask[i][j] = 1
                budget-=1
        R = Rstar * mask

        Score = []
        for i in range(No):
            sumr = 0
            cnt =0
            for j in range(Na):
                sumr+=R[i][j]
                if(R[i][j] > 0):
                    cnt+=1
            Score.append(sumr/cnt)
    
        predRank = np.argsort(Score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))
 
    print(f"Quality for Blk_rand_Lc_Bgt = {np.mean(quality_each_epoch)}")
    print(f"nDCG for Blk_rand_Lc_BgtC = {np.mean(nDCG_each_epoch)*100}")
    print("----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100


def Blk_comp(Rstar,dataset):
    """
    BASELINE - 5
    It takes the Rstar and the dataset as input and returns the mean quality@25 and mean nDCG@25 for 100
    epochs
    
    :param Rstar: The true Ground Truth Data matrix
    :param dataset: The dataset to be used
    :return: The mean of the quality@25 and nDCG@25 for each epoch.
    """
    mask = blockMask(dataset)
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        import random
        gray()
        device = 'cuda'

        Rstar_tensor = torch.from_numpy(Rstar)
        mask_tensor = torch.from_numpy(mask).to(device)

        X,Z = nncComplete(Rstar_tensor,mask_tensor)
        X = X.to('cpu')
        Z = Z.to('cpu')
        # print(f"True Rank : {linalg.matrix_rank(X)}")
        # print(f"Pred Rank : {linalg.matrix_rank(Z)}")

        tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        Score = torch.sum(Z,dim=1).numpy()
        trueRank = np.argsort(tr_score)[::-1]
        predRank = np.argsort(Score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))
    

    print(f"Quality for Blk_comp = {np.mean(quality_each_epoch)}")  
    print(f"nDCG for Blk_comp = {np.mean(nDCG_each_epoch)*100}")   
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100

def Blk_rand_Bgt_comp(Rstar,dataset,No,Na,extraBudget=100):
    """
    BASELINE - 6
    
    :param Rstar: The Ground Truth Data Matrix
    :param dataset: The dataset to be used
    :param No: Number of objects
    :param Na: Number of Assessor
    :param extraBudget: The extra budget that we are adding to the original budget, defaults to 100
    (optional)
    :return: Quality@25 and nDCG@25
    """
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        import random
        gray()
        device = 'cuda'
        mask = blockMask_with_randomQuery(dataset,extraBudget,No,Na)
        R = Rstar * mask
        Rstar_tensor = torch.from_numpy(Rstar)
        mask_tensor = torch.from_numpy(mask).to(device)

        X,Z = nncComplete(Rstar_tensor,mask_tensor)
        X = X.to('cpu')
        Z = Z.to('cpu')
        # print(f"True Rank : {linalg.matrix_rank(X)}")
        # print(f"Pred Rank : {linalg.matrix_rank(Z)}")

        trueScore = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        predScore = torch.sum(Z,dim=1).numpy()
        trueRank = np.argsort(trueScore)[::-1]
        predRank = np.argsort(predScore)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))


    print(f"Quality for Blk-rand-Bgt-comp = {np.mean(quality_each_epoch)}")    
    print(f"nDCG for Blk-rand-Bgt-comp = {np.mean(nDCG_each_epoch)*100}")    
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100


def Blk_Lc_Bgt_comp(Rstar,No,Na,dataset,rank=3,extraBudget=100):
    """
    BASELINE - 7
    
    :param Rstar: The true ground truth matrix
    :param No: Number of objects
    :param Na: Number of assessors
    :param dataset: The dataset to be used
    :param rank: Rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The extra budget that we are going to use for Active Query, defaults to 100
    (optional)
    :return: Quality@25 and nDCG@25
    """
    import random
    gray()
    device = 'cuda'
    mask = blockMask(dataset)
    R = Rstar * mask
    c0 = 0.1
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
        R_copy = R.copy()
        mask_copy = mask.copy()
        budget = extraBudget
        while(budget):
            i = np.random.randint(No)
            j = np.random.randint(Na)
            while(mask_copy[i][j] == 1):
                i = np.random.randint(No)
                j = np.random.randint(Na)
            probability = P_LC[i][j]
            shouldSample = coinFlip(probability)
            if(shouldSample):
                mask_copy[i][j] = 1
                budget-=1


        Rstar_tensor = torch.from_numpy(Rstar)
        mask_tensor = torch.from_numpy(mask_copy).to(device)

        X,Z = nncComplete(Rstar_tensor,mask_tensor)
        X = X.to('cpu')
        Z = Z.to('cpu')
        # print(f"True Rank : {linalg.matrix_rank(X)}")
        # print(f"Pred Rank : {linalg.matrix_rank(Z)}")
        tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        pr_score = torch.sum(Z,dim=1).numpy()
        truerank = np.argsort(tr_score)[::-1]
        predRank = np.argsort(pr_score)[::-1]
        quality_at_k = Calculate_Quality_at_k(truerank,predRank)

        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,truerank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))

    print(f"Quality for Blk_Lc_Bgt_comp = {np.mean(quality_each_epoch)}")
    print(f"nDCG for Blk_Lc_Bgt_comp = {np.mean(nDCG_each_epoch)*100}") 
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100

def Blk_rand_Lc_Bgt_comp(Rstar,No,Na,dataset,rank=3,extraBudget=100):
    """
    BASELINE - 8

    :param Rstar: The true Ground truth matrix
    :param No: Number of objects
    :param Na: Number of assessor
    :param dataset: The dataset to be used
    :param rank: The rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The extra budget that we have to spend on Active Query, defaults to 100 (optional)
    :return: Quality@25 and nDCG@25
    """
    import random
    gray()
    device = 'cuda'
    c0 = 0.1
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        mask = blockMask_with_randomQuery(dataset,extraBudget/2,No,Na)
        R = Rstar * mask

        P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
        budget = extraBudget/2
        while(budget):
            i = np.random.randint(No)
            j = np.random.randint(Na)
            while(mask[i][j] == 1):
                i = np.random.randint(No)
                j = np.random.randint(Na)
            probability = P_LC[i][j]

            shouldSample = coinFlip(probability)
            if(shouldSample):
                mask[i][j] = 1
                budget-=1


        Rstar_tensor = torch.from_numpy(Rstar)
        mask_tensor = torch.from_numpy(mask).to(device)

        X,Z = nncComplete(Rstar_tensor,mask_tensor)
        X = X.to('cpu')
        Z = Z.to('cpu')
        # print(f"True Rank : {linalg.matrix_rank(X)}")
        # print(f"Pred Rank : {linalg.matrix_rank(Z)}")

        tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        pr_score = torch.sum(Z,dim=1).numpy()
        trueRank = np.argsort(tr_score)[::-1]
        predRank = np.argsort(pr_score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)
        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance,trueRelevance))


    print(f"Quality for Blk_rand_Lc_Bgt_comp = {np.mean(quality_each_epoch)}")
    print(f"nDCG for Blk_rand_Lc_Bgt_comp= {np.mean(nDCG_each_epoch)*100}")
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100

def OPLP_Query(Rstar,No,Na,dataset,variant= "AfterBgtComp",rank=3,extraBudget=100):
    """
    It takes a dataset, and a variant of OPLP-Query, and returns the average quality and nDCG over 100
    epochs.
    
    :param Rstar: The true ground truth matrix
    :param No: Number of objects
    :param Na: Number of assessor
    :param dataset: The dataset to be used
    :param variant: "AfterBgtComp" or "withBgtComp", defaults to AfterBgtComp (optional)
    :param rank: rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The extra budget to be used for querying, defaults to 100 (optional)
    :return: the mean quality@25 and mean nDCG@25 for the OPLP-Query algorithm.
    """
    import random
    gray()
    device = 'cuda'
    c0 = 0.1
    quality_each_epoch = [] 
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epo in tqdm(range(100)):
        mask = blockMask(dataset)
        R = Rstar * mask
        Z = R.copy()
        Z = torch.from_numpy(Z)
        P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
        budget = extraBudget
        while(budget): 
            Zn = Z.numpy().copy()
            for i in range(Zn.shape[0]):
                for j in range(Zn.shape[1]):
                    if(Zn[i][j] < 0 ):
                        Zn[i][j] = 0

            Score = []
            for i in range(No):
                sumr = 0
                cnt =0
                for j in range(Na):
                    sumr+=Zn[i][j]
                    if(Zn[i][j] > 0):
                        cnt+=1

                Score.append(sumr/cnt)

            P_obj = Score / sum(Score)
            row = np.random.choice(No,1,p=P_obj)[0]
            P_LC = driver(rank,Rstar,R,mask,c0,No,Na)
            P_asr = P_LC[row] / sum(P_LC[row])
            col = np.random.choice(Na,1,p=P_asr)[0]
            if mask[row][col] == 0:
                mask[row][col] = 1
                budget-=1
                R = Rstar * mask
                if variant == "AfterBgtComp":
                    Z = R.copy()
                    Z = torch.from_numpy(Z)
                else:
                    Rstar_tensor = torch.from_numpy(Rstar)
                    mask_tensor = torch.from_numpy(mask).to(device)
                    
                    X,Z = nncComplete(Rstar_tensor,mask_tensor)
                    X = X.to('cpu')
                    Z = Z.to('cpu')
  
        
        Rstar_tensor = torch.from_numpy(Rstar)
        mask_tensor = torch.from_numpy(mask).to(device)
        X,Z = nncComplete(Rstar_tensor,mask_tensor)
        X = X.to('cpu')
        Z = Z.to('cpu')
        tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        pr_score = torch.sum(Z,dim=1).numpy()
        trueRank = np.argsort(tr_score)[::-1]
        predRank = np.argsort(pr_score)[::-1]
        quality_at_k = Calculate_Quality_at_k(trueRank,predRank)

        quality_each_epoch.append(quality_at_k)
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))
        saveRanks(predRank,"OPLP_Query",'pr')
        saveRanks(trueRank,"block",'tr')

    print(f"Quality for OPLP-Query ({variant}) = {np.mean(quality_each_epoch)}")    
    print(f"nDCG for OPLP-Query ({variant}) = {np.mean(nDCG_each_epoch)*100}")    
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100


#Noise free Algorithm Blueprint
def BFLP_Query(Rstar,No,Na,dataset,rank=3,extraBudget=100):
    """
    BFLP-Query 
    It takes a partially observed matrix, completes it using NNC, and then uses the completed matrix to
    query the next entry.
    
    :param Rstar: The true ground truth matrix
    :param No: Number of objects
    :param Na: Number of assessors
    :param dataset: The dataset to be used
    :param rank: The rank of the matrix Rstar, defaults to 3 (optional)
    :param extraBudget: The number of extra queries you want to make, defaults to 100 (optional)
    :return: Quality@25 and nDCG@25
    """
    gray()
    device = 'cuda'
    d = rank 
    quality_each_epoch = []
    nDCG_each_epoch = []
    print("\nEpoch#")
    for epoch in tqdm(range(3)):
        mask = blockMask(dataset)
        R = Rstar * mask
        for budget in range(extraBudget):
            print(f"Till Now Queried={budget}/{extraBudget}")

            Rstar_tensor = torch.from_numpy(Rstar)
            mask_tensor = torch.from_numpy(mask).to(device)

            X,Z = nncComplete(Rstar_tensor,mask_tensor)
            X = X.to('cpu')
            Z = Z.to('cpu')   #Completed Matrix
            Z = Z.numpy()
            Z_org = Z.copy()
            maxSum = 0
            for row in Z:
                if(maxSum < sum(row)):
                    maxSum = sum(row)

            S = []
            for i in  range(len(Z)):
                S.append(Z[i]/maxSum)
            Z = S.copy()
            Z = np.array(Z)



            objects = [i for i in range(No)]
            assessors = [i for i in range(Na)]
            I1 = np.random.choice(objects,d)
            J1 = np.random.choice(assessors,d)

            mask_d=np.zeros((No,Na))
            mask_d[I1]=1
            mask_d.T[J1] = 1 
            New_R = Z * mask
            I_star = I1

            combi  = list(itertools.combinations([i for i in range(No)], d))
            combi_arr = np.array(combi)

            # comb = combi_arr[0]

            for comb in combi_arr:
                A = New_R[I_star]
                A = A[:,J1]
                det_Istar = (np.linalg.det(A))**2
                I = New_R[comb]
                I = I[:,J1]
                det_I = (np.linalg.det(I))**2
                if(det_I > det_Istar):
                    I_star = comb     # [21,34,45]


            #Now get these d students and apply local coherence on them
            Reduced_mat  = R[I_star]
            rMask = mask[I_star]
            c0 = 0.1
            P_LC = driver(d,Rstar,R,mask,c0,No,Na)

            IsSampled = False
            while(IsSampled == False):
                i = np.random.randint(d)
                row = I_star[i]
                col = np.random.randint(Na)
                while(mask[row][col] == 1):
                    i = np.random.randint(d)
                    row = I_star[i]
                    col = np.random.randint(Na)
                probability = P_LC[row][col]
                shouldSample = coinFlip(probability)  
                if(shouldSample):
                    mask[row][col] = 1
                    IsSampled = True


            R = Rstar * mask
            #1. number of true d identified correctly.
            #2. whether top guy identified

            tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
            pr_score = torch.sum(torch.from_numpy(Z_org),dim=1).numpy()
            trueRank = np.argsort(tr_score)[::-1]
            predRank = np.argsort(pr_score)[::-1]
            
        tr_score = torch.sum(torch.from_numpy(Rstar),dim=1).numpy()
        pr_score = torch.sum(torch.from_numpy(Z_org),dim=1).numpy()
        trueRank = np.argsort(tr_score)[::-1]
        predRank = np.argsort(pr_score)[::-1]

        quality_each_epoch.append(Calculate_Quality_at_k(trueRank,predRank))
        trueRelevance,predRelevance = get_RelevanceScore_of_rankings(predRank,trueRank)
        nDCG_each_epoch.append(Calculate_ndcg_at_k(predRelevance, trueRelevance))

        saveRanks(predRank,"BFLP_Query",'pr')
        saveRanks(trueRank,"block",'tr')

    print(f"Quality for BFLP_Query = {np.mean(quality_each_epoch)}")     
    print(f"nDCG for BFLP_Query = {np.mean(nDCG_each_epoch)*100}")  
    print("\n----------------------------------")
    return np.mean(quality_each_epoch),np.mean(nDCG_each_epoch)*100




    