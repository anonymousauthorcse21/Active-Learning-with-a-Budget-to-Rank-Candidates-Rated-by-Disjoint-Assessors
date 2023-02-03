from util import *
def synthetic_Dataset_1_Generation(No,Na,rank):
    """
    It generates a synthetic dataset of size (No,Na) where No is the number of Objects and Na is the
    number of assessor. The function generates a random matrix F of size (No,rank) and a random matrix Wt of size (rank,Na). The final dataset is generated
    by multiplying F and Wt
    
    :param No: Number of Objects
    :param Na: Number of Assessors
    :param rank: The rank of the matrix
    """
    noofStudent = No
    noofFeature  = rank
    F = np.random.randint(1,10,(noofStudent,noofFeature))

    Wt=[]
    std  = 3
    for _ in range(rank):
        arr=[]
        i=0
        mu=np.random.randint(1,10)
        while i<Na:
            x = np.random.normal(mu,std)
            if (x>0.0):
                arr.append(x)
                i += 1

        Wt.append(arr)
    Wt=np.array(Wt)
    Rstar = np.matmul(F,Wt)
    np.savetxt('Synthetic_DataSet_1(Gaussian_Weightage).csv', Rstar, delimiter=",")
    


def synthetic_Dataset_2_Generation(No,Na,rank = 3):
    """
   
    :param No: Number of objects
    :param Na: Number of assessors
    :param rank: The rank of the matrix, defaults to 3 (optional)
    """

    k = [1]*rank
    N = No-rank
    alpha = np.random.dirichlet(alpha=k, size=N)
    x = np.random.randint(1,10,(rank,rank))


    x_new = np.matmul(alpha, x)
    x_new = x_new.tolist()
    X = x.tolist() + x_new
    F = np.array(X)
    np.random.shuffle(F)

    N = Na-rank
    x = np.random.randint(1,10,(rank,rank))
    alpha1 = np.random.dirichlet(alpha=k, size=N)

    x_new = np.matmul(alpha1, x)
    x_new = x_new.tolist()
    X = x.tolist() + x_new
    W = np.array(X)
    np.random.shuffle(W)

    Rstar = F @ W.T
    np.savetxt('Synthetic_DataSet_2(Convex_Hull).csv', Rstar, delimiter=",")


