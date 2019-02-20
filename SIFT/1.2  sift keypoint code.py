import cv2
from math import e,pi,sqrt
import numpy as np

#function for padding.
def padding(grayl):
    apgray=[[0. for j in range(len(grayl[0])+6)] for i in range(len(grayl)+6)]
    for i in range(len(grayl)):
        for j in range(len(grayl[0])):
            apgray[i+3][j+3] = grayl[i][j]
    return apgray

#function to create an octave of kernels with different sigmas.                                           
def create_octave_kernels(sig):
    octave[:] =[]
    octave2[:] = []
    dup=0
    gauk=[[0 for j in range(dimker)] for i in range(dimker)]
    gauke=[[0 for j in range(dimker)] for i in range(dimker)]
    
    for s in range(5):     
        for i in range(dimker):
            for j in range(dimker):
                x = i - dimker//2
                y = j - dimker//2
                gauk[i][j]=(1/(2*pi*((sig)**2)))*(e**(-(x**2+y**2)/(2*(sig)**2)))
                dup = dup+gauk[i][j]
        sig = sig * sqrt(2)
#normalizing the gaussian kernels.
        for i in range(dimker):
            for j in range(dimker):
                gauke[i][j]=((gauk[i][j])/dup)
        octave.append(gauke)
        octave2.append(gauke)
        dup=0
        gauk=[[0 for j in range(dimker)] for i in range(dimker)]
        gauke=[[0 for j in range(dimker)] for i in range(dimker)]
        def clean_octave():
            octave[:]=[]
            return
    if(countre==1):
            clean_octave()
            return octave2
    return octave

#function for convolution with gaussian kernels.
def convolu(ker,gray):
    grayp = padding(gray) #padding the source image.
    ksf=[[0 for j in range(len(ker[0]))] for i in range(len(ker))]  
    for i in range(len(ker)):
        for j in range(len(ker)):
            ksf[i][j]=ker[len(ker)-i-1][len(ker)-j-1]  
    res=[[0. for j in range(len(gray[0]))] for i in range(len(gray))]
    x=0
    kh=len(ksf)//2
    kw=len(ksf[0])//2
    ih=len(grayp)
    iw=len(grayp[0])
    for i in range(kh,ih-kh):
        for j in range(kw,iw-kw):
            x=0
            for l in range(len(ksf)):
                for k in range(len(ksf[0])):
                    x = x+ grayp[i-kh+l][j-kw+k]*ksf[l][k]      
            res[i-3][j-3]=x
    return res 

#function to resize the image.
def reduce(imager):
    red = [[0 for a in range(len(imager[0])//2)] for b in range(len(imager)//2)]
    for i in range(1, (len(imager)//2)-1):
        for j in range(1, (len(imager[0])//2)-1):
            red[i][j] = imager[2*i][2*j]
    return red  
    
#function to store all the convolved images in the gau_oct.   
def octavemat(octave):
    gau_oct[:]=[]
    for i in range(len(octave)):
        resl = convolu(octave[i],gray)
        gau_oct.append(resl)
        cv2.namedWindow('octave'+str(octave_num+1)+'.'+str(i+1)+' size = '+str(len(resl))+'x'+str(len(resl[0])),cv2.WINDOW_NORMAL)
        cv2.imshow('octave'+str(octave_num+1)+'.'+str(i+1)+' size = '+str(len(resl))+'x'+str(len(resl[0])),np.asarray(resl))
    return gau_oct
        
#function to find the difference of gaussians.
def diff_of_gau(gau_oct):
    diff_gau[:]=[]
    
#function to elimintae the zeros.
    def eliminate2(resl):
        for i in range(len(resl)):
            resl[i]=[abs(j) for j in resl[i]]
        maximum = max([max(j) for j in resl])
        for i in range(len(resl)):
            if maximum !=0:
                resl[i][:] = [x / maximum for x in resl[i]]
        return resl
    diff=[[0 for t in range(len(gau_oct[0][0])+1)] for u in range(len(gau_oct[0])+1)]
#finding the difference of gaussians
    for k in range(1,len(gau_oct)):
        for i in range(len(gau_oct[0])):
            for j in range(len(gau_oct[0][0])):
                diff[i][j] = gau_oct[k][i][j]-gau_oct[k-1][i][j]
        diff = eliminate2(diff)
        diff_gau.append(diff)
        cv2.namedWindow('diff'+str(octave_num+1)+'.'+str(k),cv2.WINDOW_NORMAL)
        cv2.imshow('diff'+str(octave_num+1)+'.'+str(k),np.asarray(diff))
        diff=[[0 for t in range(len(gau_oct[0][0])+1)] for u in range(len(gau_oct[0])+1)]
    print("diff is over")
    return diff_gau

#function to project the keypoints onto the original image.
def projection(tup,imager):
    for co in tup:
        imager[co[0]][co[1]] = (255,255,255)
    return imager

#function to find all the keypoints for DOG 1,2,3.
def keypoint2(diff_gau):
    key=[[0 for t in range(len(gau_oct[0][0])+1)] for u in range(len(gau_oct[0])+1)]
    dh = len(diff_gau[0])
    dw = len(diff_gau[0][0])
    
    for i in range(1,dh-1):
        for j in range(1,dw-1):
            b1 = 0
            c1 = 0
            for l in range(3):
                for k in range(3):
                    b = max(diff_gau[1][i-1+l][j-1+k],diff_gau[2][i-1+l][j-1+k],diff_gau[3][i-1+l][j-1+k])
                    c = min(diff_gau[1][i-1+l][j-1+k],diff_gau[2][i-1+l][j-1+k],diff_gau[3][i-1+l][j-1+k])
                    if(b>=b1):
                        b1=b
                    if(c<=c1):
                        c1=c
            
            if(diff_gau[2][i][j]>=b1):
                key[i][j]=2*diff_gau[2][i][j]
                

                tup.append(((2**octave_num)*i,(2**octave_num)*j))
                key[i][j]=2*diff_gau[2][i][j]

            elif(diff_gau[2][i][j]<=c1):
                key[i][j]=2*diff_gau[2][i][j]
#                if(i > (octave_num+1)*3 and i<(len(gray)-((octave_num+1)*3)) and j > (octave_num+1)*3 and j<(len(gray)-((octave_num+1)*3))):
                tup.append(((2**octave_num)*i,(2**octave_num)*j))
                key[i][j]=2*diff_gau[2][i][j]
                
                        
    cv2.namedWindow('keypoint for 1,2,3'+str( octave_num+1),cv2.WINDOW_NORMAL)
    cv2.imshow('keypoint for 1,2,3'+str( octave_num+1),np.asarray(key))
    return

#function to find all the keypoints fro DOG 0,1,2.
def keypoint1(diff_gau):
    key=[[0 for t in range(len(gau_oct[0][0])+1)] for u in range(len(gau_oct[0])+1)]
    dh = len(diff_gau[0])
    dw = len(diff_gau[0][0])

    for i in range(1,dh-1):
        for j in range(1,dw-1):
            b1 = 0
            c1 = 0
            for l in range(3):
                for k in range(3):
                    b = max(diff_gau[0][i-1+l][j-1+k],diff_gau[1][i-1+l][j-1+k],diff_gau[2][i-1+l][j-1+k])
                    c = min(diff_gau[0][i-1+l][j-1+k],diff_gau[1][i-1+l][j-1+k],diff_gau[2][i-1+l][j-1+k])
                    if(b>=b1):
                        b1=b
                    if(c<=c1):
                        c1=c
            
            if(diff_gau[1][i][j]>=b1):
                key[i][j]=2*diff_gau[1][i][j]   
                tup.append(((2**octave_num)*i,(2**octave_num)*j))
                key[i][j]=2*diff_gau[1][i][j]
              
            elif(diff_gau[1][i][j]<=c1):
                key[i][j]=2*diff_gau[1][i][j]               
                tup.append(((2**octave_num)*i,(2**octave_num)*j))
                key[i][j]=2*diff_gau[1][i][j]

                        
    cv2.namedWindow('keypoint for 0,1,2'+str(octave_num+1),cv2.WINDOW_NORMAL)
    cv2.imshow('keypoint for 0,1,2'+str(octave_num+1),np.asarray(key))
    return
    

    
dimker = 7
gauk=[[0 for j in range(dimker)] for i in range(dimker)]
imag = cv2.imread('c:\\task2.JPG',0)
imager = cv2.imread('c:\\task2.JPG')
imag = imag/255.0

tup = []
octave =[]
octave2=[]
gau_oct=[]
diff_gau=[]
countre=0
dup=0
octave_numr = 1

for i in range(4):    
    octave_num = octave_numr-1
    if(octave_num==0):
        sig = 1/sqrt(2)
    else:
        sig = (1/sqrt(2))*octave_num*2    
    if (octave_num == 0):
        gray = imag
#extracting the 3rd image from the previous octave for image resizing.
    elif(octave_num>0):
        for i in range(octave_numr-1):
            if(i ==0):
                sigre = 1/sqrt(2)
            else:
                sigre = (1/sqrt(2))*2*i
            countre=1
            octave2 = create_octave_kernels(sigre)
            countre=0
            gray = convolu(octave2[2],gray)
            gray = reduce(gray)
            gray = gray
        
#calling all functions.        
    octave = create_octave_kernels(sig) 
    gau_oct = octavemat(octave)
    print("octave number"+"="+str(octave_numr))
    print("size = "+str(len(gau_oct[0]))+"x"+str(len(gau_oct[0][0])))                    
    diff_gau=diff_of_gau(gau_oct)
    keypoint2(diff_gau)
    keypoint1(diff_gau)
    octave_numr +=1
detect = projection(tup,imager)
cv2.namedWindow('detect',cv2.WINDOW_NORMAL)
cv2.imshow('detect',np.asarray(imager))
print("over")
cv2.waitKey(0)
cv2.destroyAllWindows()



    





                
    