from util import *

name = ["Blk-SA","Blk-rand-Bgt","Blk-Lc-Bgt","Blk-rand-Lc-Bgt","Blk-comp",\
"Blk-rand-Bgt-comp","Blk-Lc-Bgt-comp","Blk-rand-Lc-Bgt-comp","OPLP-Query(AfterBgtComp)",\
"OPLP-Query(withBgtComp)","BFLP-Query"]


def allAlgoplot(name,list,clr,w,plotName):
    plt.plot(name,list,marker = "o",linestyle="dashed",color=clr)
    plt.xticks(rotation=90)
    if w == 0:
        plt.ylabel("Quality@25(%)",fontsize=12)
    else:
        plt.ylabel("nDCG@25",fontsize=12)
    plt.savefig(plotName+".pdf",bbox_inches="tight")
                


def plot_tr_vs_pr(trPath,prPath,No=200):
  tr = np.loadtxt(trPath, delimiter =',')
  pr = np.loadtxt(prPath, delimiter =',')
  prvstr_r = []
  for i in tr:
    prvstr_r.append(list(pr).index(i))
  plt.plot([i for i in range(0,No)], prvstr_r[:No],label="BFLP-Query",color="maroon")

  plt.xlabel("True Rank",fontsize=13)
  plt.ylabel("Predicted Rank",fontsize=13)
  plt.legend(fontsize=12)
  plt.savefig("tr_vs_pr_"+prPath+"_.pdf",bbox_inches="tight")
  plt.show()



