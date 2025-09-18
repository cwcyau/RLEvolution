batch_size=30
#
#during training
#data are sampled backwards
#when step==0, it means it is the last step for the trajectory
#and step++ to make CNP more complex
def Sample_train_data(first_step_flag=True,state=None,next_state=None,advantage=None,Chr=None,step=None,wgd=None,valid=None):
    #Sample data for training (similar to the case when a machine is playing a game against itself)
    #Thus, we don't need real world data during training, as long as the reward is similar to the real world probability
    #As in theorem 1, there is no specific destribution required to compute expectation over (s,a) pairs
    #Any distribution with broad support over all (s,a) will do the job
    if first_step_flag:
        #The first sample
        state=torch.ones(batch_size,1,num_chromosome,chrom_width,requires_grad=False)
        next_state=torch.ones(batch_size,1,num_chromosome,chrom_width,requires_grad=False)
        Chr=torch.ones(batch_size,requires_grad=False).type(torch.LongTensor)
        step=torch.zeros(batch_size,requires_grad=False)
        advantage=torch.zeros(batch_size)
        wgd=torch.zeros(batch_size,requires_grad=False)
        valid=torch.ones(batch_size,requires_grad=False)
    
    #sample starting point, end point, gain or loss  
    #because of the permutation invariant structure in section 3.3.3
    #it is not necessary to resample the chromosome everytime
    start_loci=torch.randint(high=chrom_width,size=(batch_size,),requires_grad=False)
    end_loci=torch.LongTensor(batch_size)
    cnv=torch.ones(batch_size,requires_grad=False)
    chrom=torch.Tensor(batch_size,chrom_width)
    chrom_new=torch.Tensor(batch_size,chrom_width)
    #probability of resetting the training trajectory back to step=0
    step_prob=0.18+0.8/(1+math.exp(-1e-2*counter_global+2))
    for i in range(batch_size):
        #if the model is poorly trained until the current step
        #go back to the state 0
        #to ensure small error for short trajectories
        if(torch.rand(1)[0]>step_prob or torch.abs(advantage[i])>=30 or wgd.sum()>24 or step[i]>90):
            state[i]=torch.ones(1,num_chromosome,chrom_width,requires_grad=False)
            next_state[i]=torch.ones(1,num_chromosome,chrom_width,requires_grad=False)
            step[i]=0
            wgd[i]=0
        #if model is fully trained for the current step
        #and there is no invalid operations been sampled
        #go to next step
        elif(valid[i]>0 and torch.abs(advantage[i])<10):
            next_state[i]=state[i].clone()
            step[i]=step[i]+1
        #stay to further train the current step
        #or resample another action
        else:
            state[i]=next_state[i].clone()
    
        #reset advantage and valid after they have been checked
        advantage[i]=0
        valid[i]=1
        end_loci[i]=1+torch.randint(low=start_loci[i],high=50,size=(1,))[0]
        #change the chromosone that CNV is on with some probability
        #otherwise, all CNV will be on the same chromosome
        Chr[i]=torch.randint(high=num_chromosome,size=(1,))[0]
        #adding probability to sample whole chromosomal changes during training
        if torch.rand(1)[0]>0.8:
            start_loci[i]=0
            end_loci[i]=chrom_width
        #adding probability to sample losses starting from the start of chromosome
        if torch.rand(1)[0]>0.3:
            cnv[i]=0
        #increasing the probability to sample WGD during training
        prob_wgd=0.1/(1+math.exp(-step[i]+15))
        #starting to modify state and next state
        #extract preprocessing data
        #wgd          
        if (torch.rand(1)[0]<prob_wgd and wgd[i]<1) or (sum(wgd)<5):
            wgd[i]=1
            state[i]=state[i]*2
            next_state[i]=next_state[i]*2
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
        if(torch.any(state[i][0][Chr[i]][(start_loci[i])]< -0.5)):
            valid[i]=0
        #not joining breakpoints
        if(start_loci[i]>0.5 and torch.abs(chrom[i][start_loci[i]]-chrom[i][start_loci[i]-1])<0.5):
            valid[i]=0
        if(end_loci[i]<chrom_width-0.5 and torch.abs(chrom[i][end_loci[i]-1]-chrom[i][end_loci[i]])<0.5):
            valid[i]=0
        if cnv[i]>0.5 and (torch.any(state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]< 0.5)):
            valid[i]=0
    return state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid

def Modify_data(state,chrom,Chr,valid,cnv_max,model,sigma):
    #Modify the training data to train the Q values for the best action
    #place takers
    #make sure they are of correct tensor types
    #make sure they are meaningful values to avoid inf if they are not valid samples
    #otherwise nan may be generated
    start_loci=torch.randint(high=chrom_width,size=(batch_size,),requires_grad=False)
    end_loci=start_loci.clone()
    cnv=torch.ones(batch_size,requires_grad=False)
    next_state=state.clone()
    chrom_new=chrom.clone()
    advantage=torch.zeros(batch_size)
    for i in range(batch_size):
        #only deal with valid samples
        if valid[i]>0.5:
            start_loci[i]=cnv_max[i]//2
            cnv[i]=cnv_max[i]-start_loci[i]*2
            #update chrom_new
            chrom_new[i][(start_loci[i]):]=chrom_new[i][(start_loci[i]):]+(cnv[i]-0.5)*2
    
    end_loci=model.find_end(chrom,chrom_new,sigma,start_loci,cnv,valid)
    
    for i in range(batch_size):
        next_state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]=next_state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]+(cnv[i]-0.5)*2
      
      
    return state,next_state,chrom,chrom_new,cnv,start_loci,end_loci,advantage