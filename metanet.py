import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os 
import tabulate
import matplotlib
from matplotlib import pyplot as plt
import torch
torch.manual_seed(0)
from dataset import *
from uutils import *
from track import *

class meta_detector(nn.Module):
    
    def __init__(self,insize):
        super(meta_detector, self).__init__()

        self.lstm1=torch.nn.LSTM(insize, 512, 1, batch_first=True)
        self.lstm2=torch.nn.LSTM(512, 512, 1, batch_first=True)
        self.lstm3=torch.nn.LSTM(512, 512, 1, batch_first=True)
        
        self.fc1 = torch.nn.Linear(512, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 128, bias=True)
        self.fc3 = torch.nn.Linear(128, 2, bias=True)
        
        
    def forward(self,input_):
        
        ## Input shape: (batch, sequence, hidden size) 
        ishape = input_.shape
        #print(ishape)
        
        output, (h1,c1) = self.lstm1(input_)

        output, (h1,c1) = self.lstm2(output[:,-4:,:])

        output, (h1,c1) = self.lstm2(output[:,-2:,:])
        
        fc1 = self.fc1(output[:,-1,:])
        fc1 = nn.ReLU(inplace=True)(fc1)
        fc2 = self.fc2(fc1)
        
        fc2 = nn.ReLU(inplace=True)(fc2)
       
        return (torch.nn.Softmax()(self.fc3(fc2)))

def train(model, modelname, dataloader, testloader, testcats, tracker, datamin, datamax, loadpthtrack, savepthtrack):
    start_epoch=0
    epoch_end=1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for name, param in model.named_parameters():

        if name.startswith("lstm") == True and "bias" in name:

            n = param.size(0)
            start, end = n//4, n//2
            param.data[start:end].fill_(1.)

    for epoch in range(start_epoch, epoch_end):
        # Do learning rate decay
        #print("len data loader",len(dataloader))

        for step, datas in enumerate(dataloader):

            model.train()

            data,label=datas["data"].float().cuda(), datas["label"].long().cuda()
            
            print(data.shape, label)
            data = torch.div(torch.sub(data, datamin), torch.sub(datamax, datamin))
            #data = data - mean 
            #data = data/std
            #print(data.shape)
            scores = model(data)
            optimizer.zero_grad()
            #print((scores.argmax(axis=1)==label).sum().item())

            loss=torch.nn.functional.cross_entropy(scores, label)
            loss.backward()
            print("step,loss",step,loss)
            optimizer.step()
            
        for step,datas in enumerate(testloader):

            model.eval()

            data,label=datas["data"].float().cuda(), datas["label"].long().cuda()

            data = torch.div(torch.sub(data, datamin), torch.sub(datamax, datamin)) 

            scores = model(data)

            print((scores.argmax(axis=1)==label).sum().item())
         
        scheduler.step()
        
        ## Run the tracker on the test set and save the results. "/home/esra/metalstm/"
        
        
        #tracker.track(testcats, model, loadpthtrack, savepthtrack)
        #metrics_of_tracker_results(savepthtrack,testcats)
        PATH = "/home/esra/lstmmodel/"+modelname
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                "scheduler": scheduler.state_dict()
                }, PATH)
    
    
    
def eval_gt(model, testcats, seq_length):

# This function can be made to work for sequences that have a .npy file for the full track pass as well as for those with ground truth (gt).

	
	for cat in testcats:
	
		gtl = open("/home/esra/VOT/"+cat+"_gt/"+"groundtruth.txt").readlines()
		N=len(gtl)
		val_acc = 0
		
		with open("/home/esra/seren/person_search-master/"+cat+"gtvecs.npy", "rb") as f:

			gtvecs = np.load(f, encoding = 'bytes',allow_pickle=True)
			
		for i in range(N):
			
			repbuffer = np.zeros((1,seq_length,260))
			
			gtline = gtl[i].strip("\n").split(",")
			
			gtline = [float(i) for i in gtline]
			
			fidx = gtline[4]-1
			
			if i>=5:
			
				for j in range(seq_length):
				
					print(i-seq_length+1+j)
					repbuffer[0][j] = np.concatenate((gtvecs[int(i-seq_length+1+j)].squeeze(1), gtline[0:4]))
						
						
					
						
				with torch.no_grad():
				
					model.eval()
							
					scores = model(torch.Tensor(repbuffer).float().cuda())
					label = torch.Tensor([1]).long().cuda()
					val_acc += (scores.argmax(axis=1)==label).sum().item()
					
					
		
		print("Sequence val "+cat + " ", (val_acc+0.000001)/N)
			
def training_scheme(model, trainsl, datap, traincats, testcats, savepath):

	
	tracker = track.Tracker(seq_length)
	
	train(model, trainsl, datap, tracker, savepath, testcats)
	savepath="/home/esra/metalstmres/"
	if not os.path.exists(svpth):

		os.makedirs(svpth)
# We can run the tracker on all categories. There's no harm. It will start saving the tracking results by opening a file in the location we call savepath2 for the iterative mode.
# For the iterative set, we will only use traincats. But we also want to see the performance in testcats. So let's run it on all of them here.

	svpth = tracker.track(model, savepath, traincats+testcats)
## After the training of the Model is finished, we scaled the tracker and saved the results for all categories.

	tracker.compute_metrics(savepath, traincats+testcats)
	
	for i in range(2):
		
# It's needed to create the Save sequence dataset. That's why we only provide traincats.

		itpath, datalength = tracker.save_sequence(svpth, traincats)
		itset = VOTLTMetaMerged(itpath, datalength)
		train(model, itset, datalength, tracker, savepath, testcats)
		
		svpth = tracker.track(model, savepath, traincats+testcats)
		
	
