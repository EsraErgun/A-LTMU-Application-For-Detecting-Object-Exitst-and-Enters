from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os 

class LSTMData(Dataset):
    
# The datasets only keep the index for each sample. Therefore, it is necessary to provide the folder where the detections are located.
# Since we produce multiple types of classifier datasets for a video dataset, we also provide the folder of the merged dataset.
# sizepth: The size array is also saved in .npy format inside the file where the merged type of the dataset is saved.

    def __init__(self, pthdataset, cats, sizepth, detectionspath):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pth = pthdataset
        
        
        with open(self.pth, "rb") as f:
            
            
            self.frnod = np.load(f, encoding = 'bytes', allow_pickle= True)
            self.objids = np.load(f, encoding = 'bytes', allow_pickle= True)
            self.target = np.load(f, encoding = 'bytes', allow_pickle= True)
            
        
        self.sizepth = sizepth
        self.cats = cats
        self.detpath = detectionspath
        with open(sizepth, 'rb') as f:
            
            self.sizevec = np.load(f, encoding = 'bytes', allow_pickle= True)
        
        self.sizevec = np.cumsum(self.sizevec)
        
        self.length = self.sizevec[-1]

    def __len__(self):
        
        return self.length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sc = idx+1
        catid = (self.sizevec>=sc).nonzero()[0][0]
        
        catname = self.cats[catid]
        startingimage = self.frnod[idx]
        #print("startimage", idx, catname, self.sizevec, self.cats, startingimage)
        objs = self.objids[idx]
        
        dataholder = []
        for i in range(objs.shape[0]):
            
            
            with open(self.detpath+catname+"/"+str(int(startingimage)+i)+".npy", "rb") as f:
                
                dets = np.load(f, encoding = 'bytes', allow_pickle= True)
                sims = np.load(f, encoding = 'bytes', allow_pickle= True)
                feas = np.load(f, encoding = 'bytes', allow_pickle= True)
          

            dataholder.append(np.concatenate((feas[int(objs[i])],dets[int(objs[i])][0:4]), axis = 0))
            
                            
        targetholder = self.target[idx][0]

        ## Apply transform:

        dataholder = np.array(dataholder)



        sample = {'data': torch.Tensor(dataholder), 'label': torch.Tensor(np.array(targetholder))}


        return sample
        
def dataset_from_track_results(cats,loadpath, savepath):

# Save the tracking results in the file as sequences of 6. Separate for each sequence.
# Savepath is under the dataset folder at /home/esra/metalstm/datasets/votltfromtracking1/

    for cat in scats:

        # 
        with open(loadpath+cat+'.npy', 'rb') as f:
            
            tres = np.load(f, encoding = 'bytes', allow_pickle= True)
            trevec = np.load(f, encoding = 'bytes', allow_pickle= True)
            detobjs = np.load(f, encoding = 'bytes', allow_pickle= True)    
            decobjs = np.load(f, encoding = 'bytes', allow_pickle= True)

        gtpath = "/home/esra/VOT/"+str(cat)+"_gt/groundtruth.txt"


        gts = open(gtpath).readlines()

        dataframeno = []
        datasequence = []
        target = []

        fpc, fnc = 0,0
    
        for i in range(tres.shape[0]):

            
            if i>=6:    

                gt = gts[i+1].strip("\n").split(",") 
                frname = gt[-1]
                dataframenoo=frname
                datasequencee = decobjs[i-5:i+1]
                if gt[0]=="nan":

                    gt = [0,0,0,0]
                    
                else:

                    gt= [float(j) for j in gt]
                    gt[3]+=gt[1]
                    gt[2]+=gt[0]

                    ioul = iou(gt, list(tres[i][0:4]))

                if ioul>0.2:

                    dataframeno.append(frname)
                    datasequence.append(datasequencee)
                    target.append([1])
                else:

                    dataframeno.append(frname)
                    datasequence.append(datasequencee)

                    target.append([0])

                if ioul<0.2 and index in detobjs[i]:

                    fpc+=1

                if ioul>0.2 and index not in detobjs[i]:

                    fnc+=1

            with open(path+cat+"sequence"+'.npy', 'wb') as f:
                np.save(f, np.asarray(dataframeno))
                np.save(f, np.asarray(datasequencee))
                np.save(f, np.asarray(target)) 
