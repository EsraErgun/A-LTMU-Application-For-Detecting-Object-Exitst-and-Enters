import numpy as np
import torch 
from pickle import load
import torch
torch.manual_seed(0)
from uutils import *
import numpy as np
import torch
import os 
summation=0
## Serialize individual category arrays and normalize.
seq_length = 6
class metrics():
    
    
    def __init__(self, gtpath, predict):
        
        
        self.gtlist = open(gtpath).readlines()
        self.nanidx = []
        self.gt = []
        
        for index, el in enumerate(self.gtlist):
            
            el = el.strip("\n").split(",")
            
            if el[0]=="nan": self.nanidx.append(index)
                
            el = [float(k) for k in el]
            
            el[2]+=el[0]
            el[3]+=el[1]
            
            self.gt.append(el)
            
        self.trpr = predict
        self.N = self.trpr.shape[0]
    def iou(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def precision(self):
        
        precision = 0
        prc = 0
        for i in range(self.N):
            
            if sum(self.trpr[i])!=0 and (i+1) not in self.nanidx:
                precision+=self.iou(self.trpr[i][0:4], self.gt[i+1][0:4])
                
                print(self.trpr[i], self.gt[i+1])
                print("iou",iou(self.trpr[i][0:4], self.gt[i+1][0:4]))
                
            if sum(self.trpr[i])!=0:
               
                prc+=1
           
        print("predicted cases",prc)        
        return precision/prc
    
    def recall(self):
        
        recall = 0
        
        rec = 0
        print("frame no", self.N)
        print(self.gt.__len__())
        print(self.trpr.shape)
        for i in range(self.N):
            
            if sum(self.trpr[i])!=0 and (i+1) not in self.nanidx:
                recall+=self.iou(self.trpr[i][0:4], self.gt[i+1][0:4])
            
            if (i+1) not in self.nanidx:
                
                rec+=1
        print("how many gt", rec)
        return recall/rec
    
    
    def gt_detection_rates(self, dets, detobjs, savepth):
        
        gt_bbox_found_rate=0
        oim_gtfnd_list=[]
        lstm_gtfnd_list = []
        gt_lstm_bbox_rate=0
        N= dets.shape[0]
        
        for i in range(N):
            
            detno=dets[i].shape[0]
            
            ious = []
            
            for j in range(detno):
                
                ious.append(self.iou(dets[i][j][0:4],self.gt[i+1][0:4]))
                
            ious=np.asarray(ious)
            
            ## How many of the boxes contain gt
            
            objfnd = np.nonzero((ious>0.3))[0]
            
            oim_gtfnd_list.append(objfnd.shape[0])
            
            ## Did LSTM found these boxes? If so, with which rate? 
            print("objfnd",objfnd)
            print("detobjslstm", detobjs[i])
            bbs_common = set(list(objfnd)) & set(list(detobjs[i]))
            print("commons", bbs_common)
            
            lstm_gtfnd_list.append(bbs_common.shape[0])
            
            
            print("bbs_common",bbs_common, bbs_common.shape)
        
class Tracker():
    
    def __init__(self, seq_length, datamax, datamin):
        
        self.seq_length=seq_length
        self.datamax=datamax
        self.datamin=datamin
        
        
    def track(self, cats, model, loadpath, savepath):
## Loadpath is the location of the detections.
## Savepath is the file where the tracking results of each class are saved.

        for cat in cats:

            print("Tracking.... ",cat)
            tres = []

            trvecs = []

            detobjs = [] 
            decobjs=[]
            with open(loadpath+cat+".npy", "rb") as f:
                dets = np.load(f, encoding = 'bytes', allow_pickle= True)
                sims = np.load(f, encoding = 'bytes', allow_pickle= True)
                feas = np.load(f, encoding = 'bytes', allow_pickle= True)

            N = dets.shape[0]
            
            detbuffer = np.zeros((1,seq_length,260))

            for i in range(N):

                M = dets[i].shape[0] 
              
                if i < seq_length:

                    tres.append(dets[i][0][0:4])
                    mrgd = np.concatenate((feas[i][0],dets[i][0][0:4]), axis = 0)
                    trvecs.append(mrgd)
                    detbuffer[0][i] = mrgd
                    detobjs.append([])
                    decobjs.append(0)
                else:
                    
                    bufferrep = np.repeat(np.expand_dims(detbuffer[0][1:seq_length],0), M, axis = 0)
                    
                    mrgd = np.concatenate((feas[i],dets[i][:,0:4]), axis = 1)
                    
                    detLinp = torch.Tensor(np.concatenate((bufferrep, np.expand_dims(mrgd,1)), axis=1))
                    
                    detLinp = torch.div(torch.sub(detLinp,self.datamin), torch.sub(self.datamax,self.datamin))
                    
                    scores = model(detLinp.cuda())
                    
                   
                    objcls = torch.argmax(scores, axis = 1)
                    print("predictions",objcls)

                    objidx = np.nonzero(objcls.cpu().numpy())[0]
                    


                    if objidx.shape[0] == 0:     
                        detbuffer = np.roll(detbuffer,1, axis =0)
                        detbuffer[-1] = mrgd[0]
                        trvecs.append(mrgd[0])

                        tres.append([0,0,0,0,0])
                        detobjs.append([])
                        decobjs.append(0)
                    else:

                        objsims = sims[i][objidx]
                        #print("query sims of these objects", objsims)

                        detid = objidx[np.argmax(objsims)]
                        
                        print("detected obj id",cat, detid)

                        detinfo = mrgd[detid]

                        detbuffer = np.roll(detbuffer,1, axis = 0)

                        detbuffer[-1] = detinfo

                        trvecs.append(detinfo)
                        

                        tres.append(dets[i][detid])


                        detobjs.append(objidx)
                        
                        decobjs.append(detid)
                        
            with open(savepath+cat+'.npy', 'wb') as f:


                np.save(f, np.array(tres))
                np.save(f, np.array(trvecs))
                np.save(f, np.array(detobjs))
                np.save(f, np.array(decobjs))


                
def metrics_of_tracker_results(path, cats):
# Compute metrics can compute from any folder path that has track final decisions. /home/esra/metalstm/trackresults/
# Path here is identical to the savepath of track.
    

    for cat in cats:
        
        fi = open(path+cat+"_metrics.txt","w+")

        with open(path+cat+'.npy', 'rb') as f:
            tres = np.load(f, encoding = 'bytes', allow_pickle= True)
            trevec = np.load(f, encoding = 'bytes', allow_pickle= True)
            detobjs = np.load(f, encoding = 'bytes', allow_pickle= True)

        gtpath = "/home/esra/VOT/"+str(cat)+"_gt/groundtruth.txt"

        mtrc = metrics(gtpath, tres)

        pre,rec = mtrc.precision(), mtrc.recall()
        
        with open("/home/esra/seren/person_search-master/dataa/"+cat+'.npy', 'rb') as f:
            dets = np.load(f, encoding = 'bytes', allow_pickle= True)
            sims = np.load(f, encoding = 'bytes', allow_pickle= True)
            feas = np.load(f, encoding = 'bytes', allow_pickle= True)        
        savepth="/home/esra/metalstm/results/votlt/gtrandom/"
        #mtrc.gt_detection_rates(dets, detobjs, savepth)

        fi.write(cat+" "+ "prec:" +str(pre)+", rec: "+str(rec)+"\n")

        fi.close()
