#For real data with file1 containing path to the major copies and file2 containing path to the minor copies
#format is a txt file with 1100 rows and K columns, where K is the number of samples 
#each column contain a CNP
import numpy as np
import pandas as pd

major=np.loadtxt(file1)
minor=np.loadtxt(file2)
state=torch.zeros((major.shape[1],1,44,50)
for i in range(major.shape[1]):
    state[i][0][:22]=torch.Tensor(major[:,i]).view((22,50))
    state[i][0][22:]=torch.Tensor(minor[:,i]).view((22,50))
Chr=torch.zeros(state.shape[0],200)
CNV=torch.zeros(state.shape[0],200)
End=torch.zeros(state.shape[0],200)
state_copy=state.clone()
#Heur1
Chr,CNV,End,state=Heur1(state,Chr,CNV,End)

pd.DataFrame(Chr.numpy()).astype(int).to_csv("Chr_Heur1.csv",header=False,index=False)
pd.DataFrame(CNV.numpy()).astype(int).to_csv("CNV_Heur1.csv",header=False,index=False)
pd.DataFrame(End.numpy()).astype(int).to_csv("End_Heur1.csv",header=False,index=False)

state=state_copy.clone()
Chr=torch.zeros(state.shape[0],num_step*2)
CNV=torch.zeros(state.shape[0],num_step*2)
End=torch.zeros(state.shape[0],num_step*2)
Chr,CNV,End,state=Heur2(state,Chr,CNV,End)
#Heur2
pd.DataFrame(Chr.numpy()).astype(int).to_csv("Chr_Heur2.csv",header=False,index=False)
pd.DataFrame(CNV.numpy()).astype(int).to_csv("CNV_Heur2.csv",header=False,index=False)
pd.DataFrame(End.numpy()).astype(int).to_csv("End_Heur2.csv",header=False,index=False)

#RLEvolution
state=state_copy.clone()
Chr=torch.zeros(state.shape[0],num_step*2)
CNV=torch.zeros(state.shape[0],num_step*2)
End=torch.zeros(state.shape[0],num_step*2)
Chr,CNV,End,state=Deconvolute(model,state,Chr,CNV,End)
CNV[Chr>0.5]=CNV[Chr>0.5]+1
pd.DataFrame(Chr.numpy()).astype(int).to_csv("Chr_RL.csv",header=False,index=False)
pd.DataFrame(CNV.numpy()).astype(int).to_csv("CNV_RL.csv",header=False,index=False)
pd.DataFrame(End.numpy()).astype(int).to_csv("End_RL.csv",header=False,index=False)