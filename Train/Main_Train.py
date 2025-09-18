counter_global=0
#Model
Q_model=Q_learning()
#Load initial data
state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Sample_train_data()
#setting up optimizer
optimizer = optim.Adam(Q_model.parameters(), lr=1e-4,betas=(0.9, 0.99), eps=1e-06, weight_decay=1e-3)

#start training
while(counter_global< 3e6):
    counter_global=counter_global+1
    #load data
    state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Sample_train_data(False,state,next_state,advantage,Chr,step,wgd,valid)
    #compute advantage
    optimizer.zero_grad()
    advantage,cnv_max,sigma,temp,t2,t3=Q_model.forward(state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,valid)
    #compute loss
    loss=advantage.pow(2).mean()
    
    sigma=Q_model.switch.forward(state)
    loss+=0.1*((-wgd.view(-1,1)*torch.log(sigma)-(1-wgd.view(-1,1))*torch.log(1-sigma))*valid.view(-1,1)).mean()
    #train the model
    loss.backward()
    optimizer.step()
    #print(loss)
    
    #training with the best action
    #temp for the values both used in training and loading new data
    state_temp,next_state_temp,chrom,chrom_new,cnv,start_loci,end_loci,advantage_temp=Modify_data(state,chrom,Chr,valid,cnv_max,Q_model.End,sigma)
    #compute advantage
    optimizer.zero_grad()
    advantage,cnv_max,sigma,temp,temp2,temp3=Q_model.forward(state_temp,next_state_temp,chrom,chrom_new,Chr,cnv,start_loci,end_loci,valid)
    #compute loss
    loss=advantage.pow(2).mean()
    #print(loss)
    if(counter_global%10==9):
        print(loss,step.mean(),step.max(),wgd.sum())
    loss.backward()
    optimizer.step()
