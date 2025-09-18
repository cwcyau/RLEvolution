#simulate data for testing
def Simulate_data(batch_size=15,Number_of_step=70,no_WGD_flag=False):
    state=torch.ones(batch_size,1,num_chromosome,chrom_width,requires_grad=False)
    next_state=torch.ones(batch_size,1,num_chromosome,chrom_width,requires_grad=False)
    Chr=torch.ones(batch_size,requires_grad=False).type(torch.LongTensor)
    step=torch.zeros(batch_size,requires_grad=False)
    wgd=torch.zeros(batch_size,requires_grad=False)
    valid=torch.ones(batch_size,requires_grad=False)
    
    start_loci=torch.randint(high=chrom_width,size=(batch_size,),requires_grad=False)
    end_loci=torch.LongTensor(batch_size)
    cnv=torch.ones(batch_size,requires_grad=False)
    chrom=torch.Tensor(batch_size,chrom_width)
    chrom_new=torch.Tensor(batch_size,chrom_width)
    
    Chr_truth=torch.zeros(batch_size,Number_of_step)
    CNV_truth=torch.zeros(batch_size,Number_of_step)
    End_truth=torch.zeros(batch_size,Number_of_step)
    
    
    step_counter=0
    while(step_counter<Number_of_step):
        for i in range(batch_size):
            #reset valid after they have been checked
            valid[i]=1
            start_loci[i]=torch.randint(high=chrom_width,size=(1,))[0]
            end_loci[i]=1+torch.randint(low=start_loci[i],high=50,size=(1,))[0]
            if torch.rand(1)[0]>0.3:
                Chr[i]=torch.randint(high=num_chromosome,size=(1,))[0]
            #adding probability to sample chromosomal changes during training
            if torch.rand(1)[0]>0.8:
                start_loci[i]=0
                end_loci[i]=chrom_width
            #cnv
            if torch.rand(1)[0]>0.7:
                cnv[i]=0
            #modifying cnp
            prob_wgd=0.1/(1+math.exp(-step[i]+15))
            #wgd          
            if  (not no_WGD_flag) and ((torch.rand(1)[0]<prob_wgd or step[i]>30) and wgd[i]<1):
                wgd[i]=1
                state[i]=state[i]*2
                next_state[i]=next_state[i]*2
                Chr_truth[i][int(step[i])]=-1
                CNV_truth[i][int(step[i])]=-1
                End_truth[i][int(step[i])]=-1
                step[i]=step[i]+1
                continue
                #adding cnv effect
                #increasing copies when no wgd
                #decreasing copies when wgd
            if wgd[i]>0.5:
                cnv[i]=1-cnv[i]
            state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]=state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]-(cnv[i]-0.5)*2
            chrom[i]=state[i][0][Chr[i]][:]
            #reverse effect on chrom_new
            chrom_new[i]=state[i][0][Chr[i]][:]
            chrom_new[i][(start_loci[i]):]=chrom_new[i][(start_loci[i]):]+(cnv[i]-0.5)*2
            #not going to negative values
            if(torch.any(state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]< -0.5)):
                valid[i]=0
            #not joining breakpoints
            if(start_loci[i]>0.5 and torch.abs(chrom[i][start_loci[i]]-chrom[i][start_loci[i]-1])<0.5):
                valid[i]=0
            if(end_loci[i]<chrom_width-0.5 and torch.abs(chrom[i][end_loci[i]-1]-chrom[i][end_loci[i]])<0.5):
                valid[i]=0
            if valid[i]>0 :
                next_state[i]=state[i].clone()
                Chr_truth[i][int(step[i])]=Chr[i]+1
                CNV_truth[i][int(step[i])]=start_loci[i]*2+cnv[i]+1
                End_truth[i][int(step[i])]=end_loci[i]
                step[i]=step[i]+1
                
            #stay to further train the current step
            #or resample another action
            else:
                state[i]=next_state[i].clone()
        step_counter=step_counter+1
    for i in range(batch_size):
        temp_Chr=Chr_truth.clone()
        temp_CNV=CNV_truth.clone()
        temp_End=End_truth.clone()
        for j in range(int(step[i].item())):
            Chr_truth[i][int(step[i].item())-1-j]=temp_Chr[i][j]
            CNV_truth[i][int(step[i].item())-1-j]=temp_CNV[i][j]
            End_truth[i][int(step[i].item())-1-j]=temp_End[i][j]
    return Chr_truth,CNV_truth,End_truth,state

#Simulation Experiment
import pandas as pd
num_step=50
#For samples without WGD, set no_WGD_flag to True
Chr_truth,CNV_truth,End_truth,state=Simulate_data(batch_size=50,Number_of_step=num_step,no_WGD_flag=False)
pd.DataFrame(Chr_truth.numpy()).astype(int).to_csv("Chr_truth.csv",header=False,index=False)
pd.DataFrame(CNV_truth.numpy()).astype(int).to_csv("CNV_truth.csv",header=False,index=False)
pd.DataFrame(End_truth.numpy()).astype(int).to_csv("End_truth.csv",header=False,index=False)

Chr=torch.zeros(state.shape[0],num_step*2)
CNV=torch.zeros(state.shape[0],num_step*2)
End=torch.zeros(state.shape[0],num_step*2)
state_copy=state.clone()
Chr,CNV,End,state=Heur1(state,Chr,CNV,End)
#Heur1
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
