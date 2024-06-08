from torch.utils.data import Dataset, DataLoader
import torch

torch.manual_seed(0)

def iou(boxA, boxB):

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
    
def makedata(cats,datapath, savepath):
    
    for cat in cats:
    
    
    
        with open(datapath+cat+".npy", "rb") as f:

    		# Detections contain bounding box and confidence scores. 
            detections = np.load(f, encoding = 'bytes',allow_pickle=True)
            similarities = np.load(f, encoding = 'bytes',allow_pickle=True)
            features =np.load(f, encoding = 'bytes',allow_pickle=True)
 
   
        lines = open("/home/esra/VOT/"+cat+"_gt/"+"groundtruth.txt").readlines()
        N = detections.shape[0]
        print(N)
        data=[[],[]]
        target=[]
        quota=1
        # Res positive and negatives.
        resp=0
        resn=0
        datahs = []
        targeths=[]
        ## Compute GT overlap of each box in each frame
        seqsize = 20
        ioud = []
        fnidx = []
        fpidx = []
        frst=[]
        npratio=3
        statistics=[]
        for fid in range(int(N)):
     
            ioul=[]
            gtid = fid+1
            dets = detections[fid]

            gtbb = lines[gtid].strip("\n").split(",")
            print(cat)
            print("GT BB line", gtbb)
            gtbb = gtbb[0:4]
            if gtbb[0]!="nan":
                gtbb = [float(f) for f in gtbb]
                gtbb[3]+=gtbb[1]
                gtbb[2]+=gtbb[0]



            ss = similarities[fid]

            for (index,det) in enumerate(dets):

                prbb = det[0:4]
                if str(gtbb[0])!="nan":
                    iou_=iou(gtbb,prbb)
                    print("GT",gtbb,"prediction",prbb, iou_)

                    ioul.append(iou_)  
                else:

                    ioul.append(0)


            ioul = np.asarray(ioul)    
            ioud.append(ioul)

            fpc = (ioul<0.1)&(ss>0.7)

            if cat in ["kitesurfing1","bike11","group11", "group23", "longboard6", "person142", "person191","person192","sup1","wingsuit1","wingsuit2","wingsuit3","person21","person41","person51"] or cat.startswith("person") or cat.startswith("ballet") or cat.startswith("bicycle") or cat.startswith("skiing"):
                fnc = (ioul>0.85)&(ss<0.7)

            else:
                fnc = (ioul>0.5)&(ss<0.6)
            if cat.startswith("person5") or cat.startswith("person4") or cat.startswith("person20"):

                fnc = (ioul>0.9)&(ss<0.6)

            fpnidx = np.nonzero(fpc)[0]
            fnnidx = np.nonzero(fnc)[0]

            fpidx.append(fpnidx)
            fnidx.append(fnnidx)

        print("IOUs and collecting fp and negs are completed")

# We create these by traversing the detections array. The query is not considered here,
# but it is included in the ground truth (GT). Therefore, for the GT, we used the fr+1 index.
# For the frame name, we took the number next to the corresponding index in the GT. The real 
# frame name is in the GT, and we don't modify it (e.g., no need to add 2). It's treated as 
# the actual image name and will be used as such in the dataset.

        for fid in range(int(N)):
            poscount ,hposcount, negcount, hnegcount = 0,0,0,0
            gtid = fid+1
            gtbb = lines[gtid].strip("\n").split(",")
            framename = gtbb[4]	     	
            hsim = []
            holp =[]
            hsimp=[]
            padded=[]
            nadded=[]
            
	
            if fid>=(seqsize-1):
                beginfrname=float(framename)-seqsize+1
                
                for i in range(seqsize-1):
                    iouf = ioud[fid+1-seqsize+i]
                    molpidx=np.argmax(iouf)

                    holp.append(molpidx)
                    hsim.append(0) 



                ## Find the ones that has IoU greater or equal to 0.4. For that we need, ground truth. 
                dets, feas = detections[fid], features[fid]

                ## Argwhere will return a 2D array

                pidx = np.argwhere(ioud[fid] >= 0.5)


                nidx = np.argwhere(ioud[fid]<0.2)
                
                print(fid, pidx, nidx, ioud[fid])

                diff = pidx.shape[0]-quota

                if diff<0:

                    resp+=abs(diff)
                    if pidx.shape[0]>=1:    
                        for it in range(pidx.shape[0]):
                            holps=holp[:]
                            hsims=hsim[:]
                            #hsimps = hsimp[:]
                            did = pidx[it][0]
                            
                            

                            holps.append(did)

                            hsims.append(did)
                    
                            padded.append(did)
                            repeat = 1
                            resn+=1
                            la = [1,3]

                            if did not in fnidx[fid]:

                                repeat = 1
                                resn-=1
                                la = [1,1]
                            for i in range(repeat):
                                poscount+=1
                                data[0].append(beginfrname)
                                data[1].append(np.asarray(holps))
                                target.append(la)
                                data[0].append(beginfrname)
                                data[1].append(np.asarray(hsims))
                                target.append(la)
                                #data.append(np.asarray(hsimps))
                                print("2")
                                #target.append(la)
                            if did in fnidx[fid]:
                                hposcount+=1
                                datahs.append(np.asarray(holps))
                                targeths.append(la)
                                datahs.append(np.asarray(hsims))
                                targeths.append(la)

                else:
                    for i in range(pidx.shape[0]):

# For each representation, historysim and hmaxoverlaps will be merged and included in the data vector.

                        holps=holp[:]
                        hsims=hsim[:]
                        #hsimps=hsimp[:]
                        did = pidx[i][0]

                        holps.append(did)

                        hsims.append(did)
                        #hsimps.append(np.concatenate((features[fid][did],detections[fid][did][0:4])))
                        padded.append(did)
                        repeat = 1
                        resn+=1
                        la = [1,3]

                        if did not in fnidx[fid]:

                            repeat = 1
                            resn-=1
                            la = [1,1]
                        for i in range(repeat):
                            poscount+=1
                            data[0].append(beginfrname)
                            data[1].append(np.asarray(holps))
                            target.append(la)
                            data[0].append(beginfrname)
                            data[1].append(np.asarray(hsims))
                            target.append(la)
                            #data.append(np.asarray(hsimps))
                            #target.append(la)

                        if did in fnidx[fid]:
                            hposcount+=1
                            datahs.append(np.asarray(holps))
                            targeths.append(la)
                            datahs.append(np.asarray(hsims))
                            targeths.append(la)


                for i in range(nidx.shape[0]):

                    ## Negative leri ekle. 0 class ini
                    holps=holp[:]
                    hsims=hsim[:]
                    ## 
                    did = nidx[i][0]
                    holps.append(did)
                    hsims.append(did)
                    nadded.append(did)

                    repeat = 1
                    resp+=(repeat/npratio)
                    la=[0,2]
                    if did not in fpidx[fid]:

                        repeat = 1
                        resp-=1
                        la = [0,0]
                    for j in range(repeat):
                        negcount+=1
                        data[0].append(beginfrname)
                        data[1].append(np.asarray(holps))
                        target.append(la)
                        data[0].append(beginfrname)
                        data[1].append(np.asarray(hsims))
                        target.append(la)

                    if did in fpidx[fid]:
                        hnegcount+=1
                        datahs.append(np.asarray(holps))
                        targeths.append(la)
                        datahs.append(np.asarray(hsims))
                        targeths.append(la)



## We will keep false positives and false negatives in separate datasets and repeat the process and append as desired.

## If we repeat false positives 5 times, we'll repeat false negatives 15 times.


                ## Adding false positives
                """
                if fpidx[fid].shape[0]>0:
                    for it in range(fpidx[fid].shape[0]):

                        la = [0,2]
                        repeat = 1
                        el = fpidx[fid][it]

                        if el not in nadded:
                            resp+=1
                            holps=holp[:]
                            hsims=hsim[:]


                            holps.append(el)
                            hsims.append(el)                    
                            for j in range(repeat):
                                data[0].append(beginfrname)
                                data[1].append(np.asarray(holps))
                                target.append(la)
                                data[0].append(beginfrname)
                                data[1].append(np.asarray(hsims))
                                target.append(la)

                            nadded.append(el)    
                            hnegcount+=1
                            datahs.append(np.asarray(holps))
                            targeths.append(la)
                            datahs.append(np.asarray(hsims))
                            targeths.append(la)
                ## Adding false negatives
                if fnidx[fid].shape[0]>0:            
                    for it in range(fnidx[fid].shape[0]):
                        el = fnidx[fid][it]
                        la = [1,3]

                        repeat = 1

                        if el not in padded:
                            resn+=1
                            holps=holp[:]
                            hsims=hsim[:]

                            holps.append(el)
                            hsims.append(el)                    
                            for j in range(repeat):


                                data[0].append(beginfrname)

                                data[1].append(np.asarray(holps))
                                target.append(la)
                                data[0].append(beginfrname)
                                data[1].append(np.asarray(hsims))
                                target.append(la)
                            hposcount+=1
                            datahs.append(np.asarray(holps))
                            targeths.append(la)
                            datahs.append(np.asarray(hsims))
                            targeths.append(la)                                            
                            padded.append(el)

                ## Traverse the residuals, append as many as you can
                ## Adding residual positives
                if pidx.shape[0]>0:    
                    for it in range(pidx.shape[0]):
                        el = pidx[it][0]
                        if resp == 0:

                            break
                        la = [1,1]
                        if el not in padded:
                            holps=holp[:]
                            hsims=hsim[:]

                            holps.append(el)
                            hsims.append(el)
                            data[0].append(beginfrname)                        
                            data[1].append(np.asarray(holps))
                            target.append(la)
                            data[0].append(beginfrname)
                            data[1].append(np.asarray(hsims))
                            target.append(la)                        
                            poscount+=1
                            resp-=1
                            padded.append(el)
		 """
                ## Adding res negatives.
                """
                checkc = ["kitesurfing", "yamaha", "wingsuit", "skiing","ballet"]
                if cat in checkc:
                    for it in range(nidx.shape[0]):
                        el = nidx[it][0]
                        if resn == 0:

                            break
                        la = [0,0]
                        if el not in nadded:
                            holps=holp[:]
                            hsims=hsim[:]
                            negcount+=1
                            holps.append(np.concatenate((feas[el],dets[el][0:4])))
                            hsims.append(np.concatenate((feas[el],dets[el][0:4])))                        
                            data.append(np.asarray(holps))
                            target.append(la)
                            data.append(np.asarray(hsims))
                            target.append(la)                        
                            nadded.append(el)
                            resn-=1
                """
        frst.append([poscount ,hposcount, negcount, hnegcount])
        # Savepath i /home/esra/metalstm/datasets/votlt olmali	
        with open(savepath+"/"+cat+'.npy', 'wb') as f:
            np.save(f, np.asarray(data[0]))
            np.save(f, np.asarray(data[1]))
            np.save(f, np.asarray(target))

        print(cat, target.__len__())
        statistics.append(frst)

    return np.asarray(statistics)


        
import numpy as np
import cv2
from glob import glob


class VideoWriter():
    
    def __init__(self, cats, bboxdets, methodhandle, parentpth):
        
        self.cats = cats
        
        self.gt_list = ["/home/esra/VOT/"+cats[i]+"_gt/"+"groundtruth.txt" for i in range(len(cats))]
        
        self.bboxdets = bboxdets
        self.detections = []
        
        
        self.methodhandle = methodhandle
        
        self.parentpth = parentpth
        
        
    def visualize_result(self, img_path, detections, savepath, imname):
        
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(plt.imread(img_path))
        plt.axis("off")
        colors = ["#FF0000", "#14DCCF","#2885F4"]
  
        for i in range(len(self.methodhandle)):
            x1, y1, x2, y2 = detections[i][0:4]
            
            w = x2-x1
            h = y2-y1

            if x1 != "nan":

                try:
                    ax.add_patch(
                        plt.Rectangle(
                            (x1, y1), w, h, fill=False, edgecolor=colors[i], linewidth=3.5
                        )
                    )
                    ax.add_patch(
                        plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="white", linewidth=1)
                    )
                    ax.text(
                        20,
                        20+i*50,
                        self.methodhandle[i],
                        bbox=dict(facecolor=colors[i], linewidth=0),
                        fontsize=20,
                        color="white",
                    )
                    ax.text(
                        x1 + 5,
                        y1 - 18,
                        "{:.2f}".format(detections[i][5]),
                        bbox=dict(facecolor="#4CAF50", linewidth=0),
                        fontsize=20,
                        color="white",
                    )
                except:
                    pass
                
        plt.tight_layout()
        
        fig.savefig(savepath+imname)
        
        plt.show()
        
        plt.close(fig)
        
    def drawrsltimg(self):

        cats = self.cats
        
        self.respth = []
        
        for index,cat in enumerate(cats):   
            print(cat)
            img_path = "/home/esra/VOT/"+cat
            print(img_path)
            gallery_imgs = sorted(glob(img_path+"/000*.jpg"))
            print("gallery imgs",gallery_imgs.__len__())
            savepth = self.parentpth + cat+"_rsltimg/"
            
            self.respth.append(savepth)
            
            if not os.path.exists(savepth):
            
                os.makedirs(savepth)
            
            fno=2
          
            for i in range(len(gallery_imgs)):    
                print(i,cat)
                boxlist = [el[i] for el in self.bboxdets[index]]
                
                imname="0000"+str(fno)
                
                self.visualize_result(gallery_imgs[i],boxlist,savepth,imname)
               
                fno=fno+1
    
    def VW(self):
        # Call this function after drawrsltimg function, 
        for cat in self.cats:
                        
            res_path = self.respth[i]
            #plot_path = "/Users/esraergun/Desktop/VOT/"+class_name+"_plots/"
            result_image_names = sorted(glob(res_path+"/0000*.png"))
            
            vs = cv2.imread(result_image_names[0], cv2.COLOR_BGR2RG).shape[0:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mpeg')
            fps = 10
            out = cv2.VideoWriter("/home/esra/VOT/"+cat+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, vs)


            for img in result_image_names:

                im2 = cv2.imread(img, cv2.COLOR_BGR2RGB)
                out.write(np.uint8(im2))


            cv2.destroyAllWindows()
            out.release()        
            
            
from glob import glob
import cv2
def vwriter(cat, res_path):
    # Call this function after drawrsltimg function, 
    cat = "longboard"
    res_path = "/home/esra/VOT/resutvids"+cat+"_rsltimg"
    #plot_path = "/Users/esraergun/Desktop/VOT/"+class_name+"_plots/"
    result_image_names = sorted(glob(res_path+"/0000*.png"))

    vs = cv2.imread(result_image_names[0], cv2.COLOR_BGR2RGB).shape[0:2]

    #fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')


    fps = 30
    out = cv2.VideoWriter("/home/esra/VOT/"+cat+".mp4", fourcc, fps, (vs[1], vs[0]))


    for img in result_image_names:

        im2 = cv2.imread(img, cv2.COLOR_BGR2RGB)
        out.write(np.uint8(im2))


    cv2.destroyAllWindows()
    out.release()
    
    
   
import os
import numpy as np
import math
import numpy as np
import os 
import tabulate
import matplotlib
from matplotlib import pyplot as plt

class PrepareSeq():
    
## Detectionspath is the main folder where all detections are located. Categories are still at the top level.
## "Init" specifies where the entrances and exits are located.
## The "allframes" flag indicates whether the first frame is included in the detections. If it is, it's set to true; otherwise, it's set to false.

    def __init__(self,cat,detectionspath, savepath,allframes = False):
    
    
        
        self.cats = [cat]
        
        self.gtpaths= []
        self.dis_re = []
        self.recomputedsim = []
        self.sgmnt = []
        self.pth = detectionspath
        self.savepth = self.pth
        self.allframes = allframes
        with open("/home/esra/seren/person_search-master/"+self.cats[0]+"gtvecs.npy", "rb") as f:

            # Detections contain bounding box and confidence scores. 
            self.gtreps = np.load(f, encoding = 'bytes',allow_pickle=True)
            
        with open(self.pth+cat+".npy", "rb") as f:
        
            self.detections = np.load(f, encoding = 'bytes',allow_pickle=True)
            self.sims = np.load(f, encoding = 'bytes',allow_pickle=True)
            self.feas = np.load(f, encoding = 'bytes',allow_pickle=True)
            
        for cat in self.cats:

            self.gtpaths.append("/home/esra/VOT/"+cat+"_gt/"+"groundtruth.txt")
            
            lines = open("/home/esra/VOT/"+cat+"_gt/"+"groundtruth.txt").readlines()
            self.N = len(lines)
            dis, re = [], []
            
            dis_re = []
            for i in range(len(lines)-1):
                
                f, s = lines[i].strip("\n").split(","), lines[i+1].strip("\n").split(",")
                
                if f[0]!="nan" and s[0]=="nan":
                    ## The last frame number before the exit
                    dis.append([i+1])
                    
                elif f[0]=="nan" and s[0]!="nan":
                    ## The first frame number it enters
                    re.append([i+2])
                    
            for d, r in zip(dis, re):        
                
                dis_re.append(([d[0]], [r[0]]))
        
                
                
            self.dis_re.append(dis_re)
        self.dis_re = self.dis_re[0]
        
        ## We will add the enter and the exit as a frame number.

       
        if self.dis_re.__len__()==0:
            
            self.sgmnt.append([1, int(self.N/2)])
            self.sgmnt.append([int(self.N/2)+1, self.N])
        else:
            self.sgmnt.append([1,self.dis_re[0][0][0]])
            for i in range(len(self.dis_re)-1):

                if self.dis_re[i][1][0]<int(self.N/2) and self.dis_re[i+1][0][0]>int(self.N/2):

                    self.sgmnt.append([self.dis_re[i][1][0], int(self.N/2)])

                    self.sgmnt.append([int(self.N/2)+1,self.dis_re[i+1][0][0]])
                else:
                    self.sgmnt.append([self.dis_re[i][1][0], self.dis_re[i+1][0][0]])

            self.sgmnt.append([self.dis_re[-1][1][0], self.N])            
       
    def parseandsave(self, sort = True):
        counter = 1 
        wfl=[]
        for b, e in self.sgmnt:
            print("segments",b,e)
            if b>int(self.N/2):
                
                wf = "test"
            else:
                
                wf = "train"
            wfl.append(wf)
            detsl = []
            
            feasl = []
            simsl=[]
            gtv = self.gtreps[b-1]
            
            for frid in range(b+1,e+1):
                
## I started at b+1 because frame b is the query frame.
## Here, b and e indicate the starting frame number and ending frame number, respectively. So, to iterate through all frames, I ended at e+1.
## The first detection vectors we extracted didn't include the first frame.
## Let's adjust the subsequent vectors accordingly.

                if self.allframes == False:
                
                	fr = frid-2
                	
                else:
                	
                	fr = frid-1
                
                print(self.feas[fr].shape)
                
                sims =self.feas[fr].dot(gtv) 
                
                if sort == True:
                
                
			
                    srt = np.argsort(sims, axis = 0)[::-1]
                    feas = np.squeeze(self.feas[fr][srt], axis = 1)
                    print(feas.shape)
			
                    dets = np.squeeze(self.detections[fr][srt], axis =1)

                    sims = np.squeeze(sims[srt], axis=1)
                    
                else:
                    
                    feas = self.feas[fr]
                    print(feas.shape)
                    dets = self.detections[fr]
                    
                    sims = self.sims[fr]
                
                feasl.append(feas)

                detsl.append(dets)
                
                simsl.append(sims)
            
            with open(self.savepth+self.cats[0]+str(counter)+".npy","wb") as f:
                
                np.save(f, np.asarray(detsl))
                
                np.save(f, np.asarray(simsl))
                
                np.save(f, np.asarray(feasl))                
            
            gtpth = "/home/esra/VOT/"+self.cats[0]+str(counter)+"_gt"
            
            if not os.path.exists(gtpth):
            
                os.makedirs(gtpth)
                
            gtf = open("/home/esra/VOT/"+self.cats[0]+str(counter)+"_gt/groundtruth.txt","w+")
            
            gtlist=open(self.gtpaths[0]).readlines()

            for j in range(b,e+1):
                
                ## Hepsinin GT unu query dagil yaziyoruz. 
                print(b)
                gtf.write(gtlist[j-1].strip("\n")+","+ str(j))
                
                gtf.write("\n")
                
            counter+=1
            
        c = 1
        
        fcn = open("/home/esra/vcn.txt", "a")
        
        for j in range(len(self.sgmnt)):
            
            fcn.write(self.cats[0]+str(c))
            fcn.write(" "+wfl[j])
            fcn.write("\n")
            
            c=c+1
            
        fcn.close()  
        
def save_frames(cats, detectionspath, savepath, allframes=False):


    for cat in cats:

        with open(detectionspath+cat+".npy", "rb") as f:

            # Detections contain bounding box and confidence scores. 
            detectionsall = np.load(f, encoding = 'bytes',allow_pickle=True)
            # Similarities only contain feature representations. 
            similaritiesall = np.load(f, encoding = 'bytes',allow_pickle=True)
            featuresall =np.load(f, encoding = 'bytes',allow_pickle=True)

        if not os.path.exists(savepath+cat):
            os.mkdir(savepath+cat)

        loadgt= open("/home/esra/VOT/"+cat+"_gt/groundtruth.txt").readlines()
        
       
        for fr in range(detectionsall.shape[0]):
        
        # the first line in gt files is always the query. parsed detections start from the second frame. Therefore we set idx=fr+1.
            if allframes:
                
                idx = fr
            else:
                
                idx = fr+1
                
                
            frid=loadgt[idx].strip("\n").split(",")[-1]


            with open(savepath+cat+"/"+frid+".npy","wb") as f:

                np.save(f, np.asarray(detectionsall[fr]))

                np.save(f, np.asarray(similaritiesall[fr]))
                np.save(f, np.asarray(featuresall[fr]))
                
def merge_dataset(cats, datasetname, loadpath, savepath, seq_length):

    """
    SavePath icinde data ismi de icermelidir.
    Bu fonksiyon, sequence seklinde kaydedilmis .npy lari merge leyip tek bir veri seti haline getirir.
    Loadpath, /home/esra/metalstm/datasets/votlt
    Savepath, /home/esra/metalstm/datasets/votltmerged.npy mesela 

    """
   
    datasfrno = np.zeros((1))
    datassequence=np.zeros((20))
    labelss = np.zeros((1,2))
    # Split Type: train or test
    lengths=[]
    for cat_index, cat in enumerate(cats):
    
        with open(loadpath+cat+'.npy', "rb") as f:
        
        
        
            datafrno = np.load(f, encoding = 'bytes', allow_pickle= True)
            datasequence = np.load(f, encoding = 'bytes', allow_pickle= True)
            labels = np.load(f, encoding = 'bytes', allow_pickle= True)

            print("DEBUG", cat, datafrno.shape, datassequence.shape,datasequence.shape, labels.shape)
            
        lengths.append(datafrno.shape[0])
        
        datasfrno=np.hstack((datasfrno,datafrno))
        print(datasfrno.shape)
        datassequence=np.vstack((datassequence,datasequence))
        
        labelss=np.vstack((labelss,labels))
    

    
    with open(savepath+datasetname+"merged", 'wb') as f:
    
        np.save(f, np.array(datasfrno[1:]))
        np.save(f, np.array(datassequence[1:]))
        np.save(f, np.array(labelss[1:]))
    with open(savepath+datasetname+"samplesize", 'wb') as f:
    
        np.save(f, np.array(lengths))



