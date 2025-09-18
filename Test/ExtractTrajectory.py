#RLEvolution
def Deconvolute(model,cnp,Chr,CNV,End):
    '''
    Deconvolution samples the maximum action in a greedy way
    model:the trained Q-learning model
    cnp: the input CNP, shape: Number of CNP,1,44 (#Chr),50 (#regions for one chromosome)
    Chr,CNV,END: output tensor,shape: Number of CNP, maximum length of history
    output: for Chr: -1 indicates WGD, 0 indicates no action, 1~44 the chromosome
            for CNV: only valid if Chr is not -1 or 0
                     indicates the starting point (CNV//2) and the type of CNV (CNV%2==1 for gain and CNV%2==0 for loss)
            for End: only valid if Chr is not -1 or 0
                     indicates the end point for a CNV.
    '''
    max_step=int(Chr.shape[1])
    
    for i in range(cnp.shape[0]):
        flag=False
        current_cnp=cnp[i:(i+1)]
        sigma=model.switch(current_cnp)
        if sigma[0]<0.5:
            flag=True
        step=0
        while(step<max_step):
            #it is also possible to manually set the switch if deemed necessary
            sigma=model.switch(current_cnp)
            #hard classification
            if not flag:
                sigma=torch.ceil(sigma)
            #else:
            #    sigma=torch.zeros_like(sigma)
            #print(sigma)
            res_chrom=model.Chrom_model(current_cnp,sigma)
            #print(res_chrom)
            #find the chromosome with the maximum probability
            val,temp_Chr=res_chrom.max(1)
            temp_Chr=int(temp_Chr)
            Chr[i][step]=temp_Chr+1
            #WGD
            if (not torch.any(current_cnp-2*torch.floor(current_cnp/2)>0.5)) and torch.any(current_cnp>0.5):
                sigma_wgd=model.switch(torch.floor(current_cnp/2))
                res_chrom_wgd=model.Chrom_model(torch.floor(current_cnp/2),sigma_wgd)
                val_wgd,temp=res_chrom_wgd.max(1)
                if not flag:#val_wgd>=val and not flag:
                    val=val_wgd
                    Chr[i][step]=-1
                    CNV[i][step]=-1
                    End[i][step]=-1
                    flag=True
            #special action END
            val_end=torch.sum(torch.abs(current_cnp-1))*math.log(single_loci_loss)
            if val_end>=val:
                val=val_end
                Chr[i][step]=0
                #print(val_end)
                break
            #if WGD
            if Chr[i][step]< -0.5:
                current_cnp=(current_cnp/2).floor()
                flag=True
            #if not WGD or END
            elif Chr[i][step]>0.5:
                #find best CNV
                chrom=current_cnp[:,0,temp_Chr,:]
                last_step=-1
                if step>1 and Chr[i][step]==Chr[i][step-1]:
                    last_step=int(CNV[i][step-1].item())
                CNV[i][step]=model.CNV.find_one_cnv(chrom,sigma,last_step)
                cnv_temp=int(CNV[i][step]%2)
                start_temp=int(CNV[i][step]//2)
                #find best End
                chrom_new=chrom.clone()
                chrom_new[:,start_temp:]=chrom_new[:,start_temp:]+(cnv_temp-0.5)*2
                End[i][step]=model.End.find_one_end(chrom,chrom_new,sigma,start_temp,cnv_temp)
                #updata cnp
                #print(chrom)
                #print(start_temp,End[i][step],cnv_temp)
                
                current_cnp[:,0,temp_Chr,start_temp:int(End[i][step])]=current_cnp[:,0,temp_Chr,start_temp:int(End[i][step])]+(cnv_temp-0.5)*2
                
            step=step+1
    return Chr,CNV,End,current_cnp


#Heuristic method 1
def Heur1(cnp,Chr,CNV,End):
    max_step=int(Chr.shape[1])
    #treat each sample differently
    for i in range(cnp.shape[0]):
        current_cnp=cnp[i:(i+1)]
        current_locus=0
        step=0
        while(step<max_step and current_locus<num_chromosome*chrom_width):
            current_Chr=current_locus//chrom_width
            current_start=current_locus%chrom_width
            #no CNA
            if current_cnp[0][0][current_Chr][current_start]==1:
                current_locus+=1
                continue
            #determine if it is a loss or gain
            elif current_cnp[0][0][current_Chr][current_start]<1:
                current_cnv=1
            else:
                current_cnv=-1
            current_end=current_start
            #find the end point
            while(current_end<50):
                if current_cnp[0][0][current_Chr][current_end]!=current_cnp[0][0][current_Chr][current_start]:
                    break
                current_end+=1
            current_cnp[0,0,current_Chr,current_start:current_end]+=current_cnv
            Chr[i][step]=current_Chr+1
            CNV[i][step]=2*current_start+(current_cnv+1)//2+1
            End[i][step]=current_end
            step=step+1
    return Chr,CNV,End,current_cnp

#Heuristic 2
def Heur2(cnp,Chr,CNV,End):
    max_step=int(Chr.shape[1])
    
    for i in range(cnp.shape[0]):
        current_cnp=cnp[i:(i+1)]
        current_locus=0
        step=0
        flag=False
        #determine if this is a CNP with WGD
        if current_cnp.mean()<1.7:
            flag=True
        while(step<max_step and ((not flag) or current_locus<num_chromosome*chrom_width)):
            #all loci contain even copy number, do a reverse of WGD
            if current_locus==num_chromosome*chrom_width:
                flag=True
                current_locus=0
                current_cnp//=2
                Chr[i][step]=-1
                CNV[i][step]=-1
                End[i][step]=-1
                step+=1
                continue
            current_Chr=current_locus//chrom_width
            #determine if it is a whole chromosome change
            baseline=2
            if flag:
                baseline=1
            if  current_cnp[0][0][current_Chr].mean()>1.5*baseline:
                current_cnp[0][0][current_Chr]-=1
                Chr[i][step]=current_Chr+1
                CNV[i][step]=1
                End[i][step]=50
                current_locus=current_Chr*chrom_width
                step+=1
                continue
            elif  current_cnp[0][0][current_Chr].mean()<0.5*baseline:
                current_cnp[0][0][current_Chr]+=1
                Chr[i][step]=current_Chr+1
                CNV[i][step]=2
                End[i][step]=50
                current_locus=current_Chr*chrom_width
                step+=1
                continue
            current_start=current_locus%chrom_width
            #determine if it contains only 1 copy (no CNV) for samples without WGD
            #and if it contains even copy number for samples with WGD
            if (flag and current_cnp[0][0][current_Chr][current_start]==1) or ((not flag) and current_cnp[0][0][current_Chr][current_start]%2==0):
                current_locus+=1
                continue
            #determine if it is a gain or loss
            elif current_cnp[0][0][current_Chr][current_start]<2:
                current_cnv=1
            else:
                current_cnv=-1
            current_end=current_start
            #find end point
            while(current_end<50):
                if current_cnp[0][0][current_Chr][current_end]!=current_cnp[0][0][current_Chr][current_start]:
                    break
                current_end+=1
            current_cnp[0,0,current_Chr,current_start:current_end]+=current_cnv
            Chr[i][step]=current_Chr+1
            CNV[i][step]=2*current_start+(current_cnv+1)//2+1
            End[i][step]=current_end
            step=step+1
    return Chr,CNV,End,current_cnp
