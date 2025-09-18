import torch
import math
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#For each chromosome, we consider 50 SNP loci
chrom_width=50;
#44 normal chromosomes in human genome
num_chromosome=44


#normalisation constant to make normal_const*(\sum_i a_i) <1, so that the possibility of all CNV sum to a real value smaller than 1
normal_const=1e-5;
#probability of single locus gain/loss
single_loci_loss=normal_const*(1-2e-1);
#probability of WGD
WGD=normal_const*0.6;

#log probability of CNV
#used for calculating the distribution of p(a) when a is a focal CNV
const1=single_loci_loss
const2=1;

#Whole chromosome change probability
Whole_Chromosome_CNV=single_loci_loss/4;
Half_Chromosome_CNV=single_loci_loss/3;

max_copy=20

def Reward(Start,End):
    Start=Start.to(torch.float32)
    End=End.to(torch.float32)
    reward=torch.log(const1/(const2+End-Start))
    #chromosome changes
    for i in range(Start.shape[0]):
        #full chromosome
        if End[i]-Start[i]>chrom_width-0.5:
            reward[i]=math.log(Whole_Chromosome_CNV)
        #arm level changes
        if chrom_width-End[i]<0.5 and abs(chrom_width//2-Start[i])<1.5:
            reward[i]=math.log(Half_Chromosome_CNV)
        if Start[i]<0.5 and abs(chrom_width//2-End[i])<1.5:
            reward[i]=math.log(Half_Chromosome_CNV)
    return reward



#Q-function.py
#defining the Q-function 
#Q(s,a) in manuscript


#switch structure mentioned in section 3.3.4
#kernel sizes for convolution layers
nkernels_switch = [20,40,80]
activatiion_wgd=torch.tanh
class WGD_Net(nn.Module):
    def __init__(self):
        super(WGD_Net, self).__init__()
        #chromosome permutation invariant structure as described in section 3.3.3
        #slide for chromosome is 1 and the filter length in this dimension is also 1
        #thus, the same filter goes through all chromosomes in the same fashion
        self.conv1=nn.Conv2d(1, nkernels_switch[0], (1,3),(1,1),(0,1))
        self.conv2=nn.Conv2d(nkernels_switch[0],nkernels_switch[1] , (1,3),(1,1), (0,1))
        self.conv3=nn.Conv2d(nkernels_switch[1],nkernels_switch[2] , (1,5),(1,1), (0,0))
        self.linear=nn.Linear(nkernels_switch[2],1)
    
    def forward(self, x):
        #y=torch.clamp((F.relu(x.mean(3)-1)-0.5+0.5*F.relu(1-x.mean(3))),-1,1).sum((1,2))
        y=20*(x.mean((1,2,3))-1.5)
        y=y.reshape(x.shape[0],1).detach()
        x=x.reshape(x.shape[0],1,num_chromosome,chrom_width)
        x=F.max_pool2d(activatiion_wgd(self.conv1(x)),(1,5),(1,5),(0,0))
        x=F.max_pool2d(activatiion_wgd(self.conv2(x)),(1,2),(1,2),(0,0))
        x=(activatiion_wgd(self.conv3(x))).sum((2,3))
        x=self.linear(x)
        x=x/2
        #residule representation in x as described in section 3.3.4
        x=y/2+x
        x=torch.sigmoid(x)
        return x

#chromosome evaluation net 
#Used in Chrom_NN (which is Q_{phi_1}(s,c) in section 3.3.1)
#kernel sizes for convolution layers
nkernels_chr = [80,120,160]
activation_cnp=torch.tanh
class CNP_Val(nn.Module):
    def __init__(self):
        super(CNP_Val, self).__init__()
        self.conv1=nn.Conv2d(1, nkernels_chr[0], (1,5),(1,1),(0,2))
        self.conv2=nn.Conv2d(nkernels_chr[0],nkernels_chr[1] , (1,3),(1,1), (0,1))
        self.conv3=nn.Conv2d(nkernels_chr[1],nkernels_chr[2] , (1,3),(1,1), (0,1))
        self.conv4=nn.Conv2d(nkernels_chr[2],1, (1,5),(1,1), (0,0))
    
    def forward(self, x):
        x=F.max_pool2d(activation_cnp(self.conv1(x)),(1,3),(1,3),(0,1))
        x=F.max_pool2d(activation_cnp(self.conv2(x)),(1,2),(1,2),(0,1))
        x=F.max_pool2d(activation_cnp(self.conv3(x)),(1,2),(1,2),(0,1))
        #KL divergence is always nonpositive
        x=0.25+F.elu(self.conv4(x),0.25)
        #number of sample * 44 chromosomes
        x=x.reshape(x.shape[0],num_chromosome)
        return x

#Implemts Q_{phi_1}(s,c) in section 3.3.1
#It combines two chromosome evaluation nets mentioned above,
#with a switch structure in section 3.3.4 to form Q_{phi_1}(s,c)
class Chrom_NN(nn.Module):
    def __init__(self):
        super(Chrom_NN,self).__init__()
        #two parts of the Chrom_NN
        #NN for CNP without WGD 
        self.Val_noWGD=CNP_Val()
        #NN for CNP with WGD
        self.Val_WGD=CNP_Val()
    
    def forward(self,x,sigma):
        #probability for WGD, which is computed by switch structure
        sigma=sigma.expand(-1,num_chromosome)
        #we assume the copy number for each loci ranges from 0~9
        #for samples without WGD
        
        #y represents if a chromosome have abnormal copy numbers (positions with copy number other than 1)
        y=torch.ceil((torch.abs(x-1)).mean(3)/max_copy)
        y=y.reshape(x.shape[0],num_chromosome)
        y=y.detach()
        #Residule representation mentioned in section 3.3.4
        #the value for Q_{phi_1}(s,c) is computed as Val_no (the value estimated by the neural net, the residule part)+ y (the empirial estimation) 
        Val_no=self.Val_noWGD.forward(x)
        #chromosome with all 1 copies don't need any CNV and thus will be less likely mutated.
        Val_no=-(1-y)*math.log(single_loci_loss)*2+((y*Val_no).sum(1)).reshape(x.shape[0],1).expand(-1,num_chromosome)
        #for samples with WGD
        #it is similar to the previsou part, where z is an equivalent for y and Val_wgd is an equivalent for Val_no
        z=torch.ceil((torch.abs(x-2*(x//2))).mean(3)/max_copy)
        z=z.reshape(x.shape[0],num_chromosome)
        z=z.detach()
        Val_wgd=self.Val_WGD.forward(x)
        Val_wgd=-(1-z)*math.log(single_loci_loss)*2+(z*Val_wgd).sum(1).reshape(x.shape[0],1).expand(-1,num_chromosome)-math.log(WGD)
        
        #combine two NN with switch as defined in Section 3.3.4
        x=sigma*Val_wgd+(1-sigma)*Val_no
        x=-x
        return x



#starting point and gain or loss (defined as sp in manuscript) 
#Used in CNV_NN (which is Q_{phi_2}(s,c,sp) on section 3.3.1)
#kernel sizes for convolution layers
nkernels_CNV = [80,120,160,10]
activation_cnv=torch.tanh
class CNV_Val(nn.Module):
    def __init__(self):
        super(CNV_Val,self).__init__()
        self.conv1=nn.Conv2d(1, nkernels_CNV[0], (1,7),(1,1),(0,3))
        self.conv2=nn.Conv2d(nkernels_CNV[0],nkernels_CNV[1] , (1,7),(1,1), (0,3))
        self.conv3=nn.Conv2d(nkernels_CNV[1],nkernels_CNV[2] , (1,7),(1,1), (0,3))
        self.conv4=nn.Conv2d(nkernels_CNV[2], nkernels_CNV[3], (1,7),(1,1), (0,3))
        self.linear=nn.Linear(nkernels_CNV[3]*chrom_width,2*chrom_width-1)
    
    def forward(self,x):
        x=activation_cnv(self.conv1(x))
        x=activation_cnv(self.conv2(x))
        x=activation_cnv(self.conv3(x))
        x=activation_cnv(self.conv4(x))
        x=x.reshape(x.shape[0],nkernels_CNV[3]*chrom_width)
        x=self.linear(x)
        #number of samples* [(50 regions)*(2(gain or loss))-1] 
        #Only have 50*2-1=99 output dimensions because we fix the average these output
        #The average of them could be arbitrary because of the partitioning
        return x

#Implemts Q_{phi_2}(s,c,sp) in section 3.3.1
#It combines two CNV_Val nets mentioned above,
#with a switch structure in section 3.3.4 to form Q_{phi_2}(s,c,sp)
class CNV_NN(nn.Module):
    def __init__(self):
        super(CNV_NN,self).__init__()
        #two network setting
        self.CNV_noWGD=CNV_Val()
        self.CNV_WGD=CNV_Val()
    
    def forward(self,x,sigma):
        #as in section 3.3.4
        #y is the empirical estimation
        #Val_no is the redidule representation
        y=torch.Tensor(x.shape[0],chrom_width,2)
        y[:,:,0]=F.relu(x-1)
        y[:,:,1]=F.relu(1-x)
        y=y.reshape(x.shape[0],2*chrom_width)
        y=y[:,1:(2*chrom_width)]-y[:,0:1].expand(-1,2*chrom_width-1)
        y=-y.detach()*math.log(single_loci_loss)
        Val_no=self.CNV_noWGD.forward(x.reshape(x.shape[0],1,1,chrom_width))
        Val_no=y+Val_no
        
        z=((torch.abs(x-2*(x//2))).reshape(x.shape[0],chrom_width,1)).expand(-1,-1,2)
        z=z.reshape(x.shape[0],2*chrom_width)
        z=z[:,1:(2*chrom_width)]-z[:,0:1].expand(-1,2*chrom_width-1)
        z=-z.detach()*math.log(single_loci_loss)
        Val_wgd=self.CNV_WGD.forward(x.reshape(x.shape[0],1,1,chrom_width))
        Val_wgd=z+Val_wgd
        #switch
        x=sigma*Val_wgd+(1-sigma)*Val_no
        return(x)
    
     
    def find_one_cnv(self,chrom,sigma,last_cnv=-1):
        #used for finding the cnv during deconvolution
        #it is not used in training process
        
        res_cnv=self.forward(chrom,sigma)
        #if there is originally a break point for start
        #rule system 
        break_start=torch.zeros(chrom.shape[0],50,2,requires_grad=False)
        chrom_shift=torch.zeros(chrom.shape[0],50,requires_grad=False)
        chrom_shift[:,1:]=chrom[:,:49]
        #allow adding one copy for every breakpoint
        break_start[:,:,1]=torch.ceil(torch.abs(chrom-chrom_shift)/max_copy)
        #always allow adding one chromosone
        break_start[:,0,1]=1
        #don't allow lose one copy when copy number equalls 1
        break_start[:,:,0]=break_start[:,:,1]
        break_start[:,:,0]=break_start[:,:,0]*torch.ceil((chrom/2-0.5)/max_copy)
        break_start=break_start.reshape(chrom.shape[0],100)
        res_cnv_full=torch.zeros(chrom.shape[0],100)
        res_cnv_full[:,1:]=res_cnv
        
        if last_cnv>0: #not the first step or lossing one chromosome
            forbidden=2*last_cnv//2+(1-last_cnv%2)
            break_start[0][forbidden]=0
        #Prior_rule=break_start
        res_cnv_full=res_cnv_full+torch.log(break_start)
        #best cnv according to the current Q
        cnv_max_val,cnv_max=torch.max(res_cnv_full,1)
        #print(res_cnv_full)
        return int(cnv_max[0])
    


#end point
#Used in End_Point_NN (which is Q_{phi_3}(s,c,sp,ep) on section 3.3.1)
#kernel sizes for convolution layers
nkernels_End = [80,120,240]
activation_end=torch.tanh
class End_Point_Val(nn.Module):
    def __init__(self):
        super(End_Point_Val,self).__init__()
        self.conv1=nn.Conv2d(2, nkernels_End[0], (1,7),(1,1),(0,3))
        self.conv2=nn.Conv2d(nkernels_End[0],nkernels_End[1] , (1,7),(1,1), (0,3))
        self.conv3=nn.Conv2d(nkernels_End[1],nkernels_End[2] , (1,7),(1,1), (0,3))
        self.linear=nn.Linear(nkernels_End[2]*chrom_width,chrom_width-1)
    
    def forward(self,old,new):
        x=torch.Tensor(old.shape[0],2,1,chrom_width)
        x[:,0,0,:]=old
        x[:,1,0,:]=new
        x=x.detach()
        x=activation_end(self.conv1(x))
        x=activation_end(self.conv2(x))
        x=activation_end(self.conv3(x))
        x=x.reshape(x.shape[0],nkernels_End[2]*chrom_width)
        x=self.linear(x)
        #number of samples* [(chrom_width regions)-1] 
        #Only have chrom_width-1=49 output dimensions because we fix the average these output
        #The average of them could be arbitrary because of the partitioning
        return x
    
#Implemts Q_{phi_3}(s,c,sp,ep) in section 3.3.1
#It combines two End_Point_Val nets mentioned above,
#with a switch structure in section 3.3.4 to form Q_{phi_3}(s,c,sp,ep)
class End_Point_NN(nn.Module):
    def __init__(self):
        super(End_Point_NN,self).__init__()
        #two network setting
        self.Val_noWGD=End_Point_Val()
        self.Val_WGD=End_Point_Val()
    
    def forward(self,old,new,sigma):
        
        y=F.relu((old-1)*(old-new))
        y=y[:,1:chrom_width]-y[:,0:1].expand(-1,chrom_width-1)
        y=-y.detach()*math.log(single_loci_loss)
        Val_no=self.Val_noWGD.forward(old,new)
        Val_no=Val_no+y
        
        z=(old-2*(old//2))*(1-(new-2*(new//2)))
        z=z[:,1:chrom_width]-z[:,0:1].expand(-1,chrom_width-1)
        z=-z.detach()*math.log(single_loci_loss)
        Val_wgd=self.Val_WGD.forward(old,new)
        Val_wgd=Val_wgd+z
        #switch
        x=sigma*Val_wgd+(1-sigma)*Val_no
        return x
    
    
    def find_end(self,old,new,sigma,start_loci,cnv,valid):
        #used for finding the end during loading data
        res_end=self.forward(old,new,sigma)
        
        break_end=torch.zeros(old.shape[0],chrom_width,requires_grad=False)
        chrom_shift=torch.zeros(old.shape[0],chrom_width,requires_grad=False)
        chrom_shift[:,:49]=old[:,1:]
        #allow adding one copy for every breakpoint
        break_end[:,:]=torch.ceil(torch.abs(old-chrom_shift)/max_copy)
        #always allow adding one chromosone
        break_end[:,chrom_width-1]=1
        
        for i in range(old.shape[0]):
            #can't end before starting point
            break_end[i,:int(start_loci[i])]=0*break_end[i,:int(start_loci[i])]
            #don't allow lose one copy when copy number equalls 1
            if(cnv[i]<0.5):
                j=int(start_loci[i])+1
                while(j<chrom_width):
                    if(old[i][j]<1.5):
                        break
                    j=j+1
                break_end[i,j:chrom_width]=0*break_end[i,j:chrom_width]
        res_end_full=torch.zeros(old.shape[0],chrom_width)
        res_end_full[:,1:]=res_end
        #Prior_rule=break_end
        res_end_full=res_end_full+torch.log(break_end)
        end_max_val,end_max=torch.max(res_end_full,1)
        return end_max+1
    
    
    def find_one_end(self,old,new,sigma,start,cnv):
        #used for finding the end during deconvolution
        res_end=self.forward(old,new,sigma)
        
        break_end=torch.zeros(old.shape[0],chrom_width,requires_grad=False)
        chrom_shift=torch.zeros(old.shape[0],chrom_width,requires_grad=False)
        chrom_shift[:,:chrom_width-1]=old[:,1:]
        #allow adding one copy for every breakpoint
        break_end[:,:]=torch.ceil(torch.abs(old-chrom_shift)/max_copy)
        #always allow adding one chromosone
        break_end[:,chrom_width-1]=1
        #can't end before starting point
        break_end[0,:start]=0*break_end[0,:start]
        #don't allow lose one copy when copy number equalls 1
        if(cnv<0.5):
            j=start+1
            while(j<chrom_width):
                if(old[0][j]<1.5):
                    break
                j=j+1
            break_end[0,j:chrom_width]=0*break_end[0,j:chrom_width]
        res_end_full=torch.zeros(old.shape[0],chrom_width)
        res_end_full[:,1:]=res_end
        #Prior_rule=break_end
        res_end_full=res_end_full+torch.log(break_end)
        end_max_val,end_max=torch.max(res_end_full,1)
        end_max=int(end_max[0])
        return end_max+1
        

#combine all separate networks
#add Rule system

#calculating the softmax
#prevent inf when taking log(exp(x))
#log_exp is always gonna be between 1 and the total number of elements
def Soft_update(val1,soft1,val2,soft2):
    bias=val1.clone()
    log_exp=soft1.clone()
    set1=[torch.ge(val1,val2)]
    bias[set1]=val1[set1]
    log_exp[set1]=soft1[set1]+soft2[set1]*torch.exp(val2[set1]-val1[set1])
    set2=[torch.lt(val1,val2)]
    bias[set2]=val2[set2]
    log_exp[set2]=soft2[set2]+soft1[set2]*torch.exp(val1[set2]-val2[set2])
    return bias,log_exp



#Combine all the separate modules mentioned above
#Implementation of Q(s,a)
class Q_learning(nn.Module):
    def __init__(self):
        super(Q_learning,self).__init__()
        self.switch=WGD_Net()
        #the output refer to Q_{\phi_1}(s,c)
        self.Chrom_model=Chrom_NN()
        #the output refer to Q_{\phi_2}(s,c,sp)
        self.CNV=CNV_NN()
        #the output refer to Q_{\phi_3}(s,sp,c,ep)
        self.End=End_Point_NN()
    
    
    def forward(self,state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,valid):
        '''
        computing the final advantage(loss) used for training
        loss in Thereom1
        state: s in Q(s,a)
        next_state: s' in softmaxQ(s',a')
        Chr,cnv,end_loci: a in Q(s,a)
        chrom,chrom_new,start_loci,end_loci: intermediate results from s,a, which is preprossed to make computation faster
            e.g. chrom is CNP of the Chr(part of a) from the state(s)
            They could be seen as a mapping without parameters to learn:f(s,a)
        valid: a boolean array, indicating if a training sample is valid (e.g. have non negative copy numbers for all loci)
        '''
        
        #computing softmaxQ(s',a')
        #It is a tradition in RL that gradient does not backpropogate through softmaxQ(s',a'), but only through Q(s,a) to make convergence faster
        #there is no theoritical guarantee behind, and it is only a practical trick
        sigma_next=self.switch(next_state)
        x,y=self.Softmax(next_state,sigma_next)
        x=x+torch.log(y)
        #computing r(s,a)
        x=x+Reward(start_loci,end_loci)
        x=x.detach()
        
        #computing Q(s,a)
        sigma=self.switch.forward(state)
        if counter_global<3e6:
            sigma=sigma.detach()
        #Q_{phi_1}(s,c)
        res_chrom=self.Chrom_model.forward(state,sigma)
        
        #Q_{phi_2}(s,c,sp)
        res_cnv=self.CNV.forward(chrom,sigma)
        #if there is originally a break point for start
        #real world constraint as described in section 3.3.2
        #only allow starting points (sp) to be the break points of CNP
        break_start=torch.zeros(state.shape[0],chrom_width,2,requires_grad=False)
        chrom_shift=torch.zeros(state.shape[0],chrom_width,requires_grad=False)
        chrom_shift[:,1:]=chrom[:,:(chrom_width-1)]
        #allow adding one copy for every breakpoint
        break_start[:,:,1]=torch.ceil(torch.abs(chrom-chrom_shift)/max_copy)
        #always allow adding one chromosone
        break_start[:,0,1]=1
        #don't allow lose one copy when copy number equals 0, otherwise there is going to be negative copy numbers
        break_start[:,:,0]=break_start[:,:,1]
        break_start[:,:,0]=break_start[:,:,0]*torch.ceil((chrom/2-0.5)/max_copy)
        break_start=break_start.reshape(state.shape[0],2*chrom_width)
        res_cnv_full=torch.zeros(state.shape[0],2*chrom_width)
        res_cnv_full[:,1:]=res_cnv
        res_cnv_full=res_cnv_full+torch.log(break_start)
        
        #Q_{phi_2}(s,c,sp)-softmax(Q_{phi_2}(s,c,sp)) as described in section 3.3.1
        cnv_max_val,cnv_max=torch.max(res_cnv_full,1)
        cnv_softmax=res_cnv_full-cnv_max_val.reshape(state.shape[0],1).expand(-1,2*chrom_width)
        cnv_softmax=torch.exp(cnv_softmax).sum(1)
        x=x+cnv_max_val+torch.log(cnv_softmax)
        
        #Q_{phi_3}(s,c,sp,ep)
        res_end=self.End.forward(chrom,chrom_new,sigma)
        #if there is originally a break point for end
        #and if this is after the starting point
        #real world constraint in section 3.3.2
        break_end=torch.zeros(state.shape[0],chrom_width,requires_grad=False)
        chrom_shift=torch.zeros(state.shape[0],chrom_width,requires_grad=False)
        chrom_shift[:,:(chrom_width-1)]=chrom[:,1:]
        #allow adding one copy for every breakpoint
        break_end[:,:]=torch.ceil(torch.abs(chrom-chrom_shift)/max_copy)
        #always allow adding one chromosone
        break_end[:,chrom_width-1]=1
        for i in range(state.shape[0]):
            #can't end before starting point
            break_end[i,:int(start_loci[i])]=0*break_end[i,:int(start_loci[i])]
            #don't allow lose one copy when copy number equalls 1
            if(cnv[i]<0.5):
                j=int(start_loci[i])+1
                while(j<chrom_width):
                    if(chrom[i][j]<1.5):
                        break
                    j=j+1
                break_end[i,j:chrom_width]=0*break_end[i,j:chrom_width]
            
        res_end_full=torch.zeros(state.shape[0],chrom_width)
        res_end_full[:,1:]=res_end
        
        #real world constraint described in section 3.3.2
        res_end_full=res_end_full+torch.log(break_end)
        end_max_val,end_max_temp=torch.max(res_end_full,1)
        end_softmax=res_end_full-end_max_val.reshape(state.shape[0],1).expand(-1,chrom_width)
        end_softmax=torch.exp(end_softmax).sum(1)
        #Q_{phi_3}(s,c,sp,ep)-softmax(Q_{phi_3}(s,c,sp,ep)) as described in section 3.3.1
        x=x+end_max_val+torch.log(end_softmax)
        
        for i in range(state.shape[0]):
            if valid[i]>0.5:#check validity to prevent inf-inf which ends in nan
                x[i]=x[i]-res_chrom[i][int(Chr[i])]
                cnv_rank=int(start_loci[i]*2+cnv[i])
                x[i]=x[i]-res_cnv_full[i][cnv_rank]
                end_rank=int(end_loci[i]-1)
                x[i]=x[i]-res_end_full[i][end_rank]
        
        #remove training data which include invalid actions
        x=x*valid
        #return avdantage as well as a best cnv and sigma used for generating training data
        #used for training in the next step
        return x,cnv_max,sigma,res_chrom,res_cnv_full,res_end_full
     
    def Softmax(self,next_state,sigma):
        #compute softmax_{a'} Q(s',a')
        x=self.Chrom_model.forward(next_state,sigma)
        max_chrom=torch.max(x,1)[0]
        softmax_chrom=x-max_chrom.reshape(x.shape[0],1).expand(-1,num_chromosome)
        softmax_chrom=torch.exp(softmax_chrom).sum(1)
        #special action END
        #all the remaining abnormal loci are treated to be caused by several independent single locus copy number changes
        end_val=torch.sum(torch.abs(next_state-1),(1,2,3))*math.log(single_loci_loss)
        max_chrom,softmax_chrom=Soft_update(max_chrom,softmax_chrom,end_val,torch.ones(x.shape[0]))
        #if there is a WGD followed immediately
        for i in range(x.shape[0]):
            #real world constraint as described in section 3.3.2
            #do not allow (reversing) WGD when the CNP contain odd numbers for some loci
            if (not torch.any(next_state[i]-2*torch.floor(next_state[i]/2)>0.5)) and torch.any(next_state[i]>0.5):
                sigma_wgd=self.switch(torch.floor(next_state[i:(i+1)]/2))
                sigma_wgd=sigma_wgd.detach()
                wgd_val,wgd_soft=self.Softmax(torch.floor(next_state[i:(i+1)]/2),sigma_wgd)
                max_chrom[i],softmax_chrom[i]=Soft_update(torch.ones(1)*max_chrom[i],torch.ones(1)*softmax_chrom[i],torch.ones(1)*wgd_val,torch.ones(1)*wgd_soft)
        
        return max_chrom,softmax_chrom
