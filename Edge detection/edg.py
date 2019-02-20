import cv2
import numpy as np

img = cv2.imread('D:\\hough.jpg',0) #read image as as gray
#img = img/255.0 # convert pixel values to float

gray = img 

# define gaussian kernels
ksv=[[1,0,-1],[2,0,-2],[1,0,-1]]
ksh=[[1,2,1],[0,0,0],[-1,-2,-1]]


# function for convolution.
def convolu(ker):
    ksf=[[0 for j in range(len(ker[0]))] for i in range(len(ker))]  
    
#flipping the kernel
    for i in range(len(ker)):
        for j in range(len(ker[0])):
            ksf[i][j]=ker[len(ker)-i-1][len(ker)-j-1] 
            
#convolution            
    res=[[0 for j in range(len(gray[0]))] for i in range(len(gray))]
    kh=len(ksf)//2
    kw=len(ksf[0])//2
    ih=len(gray)
    iw=len(gray[0])
    for i in range(kh,ih-kh):
        for j in range(kw,iw-kw):
            x=0
            for l in range(len(ksf)):
                for k in range(len(ksf[0])):
                   x = x+ gray[i-kh+l][j-kw+k]*ksf[l][k]      
            res[i][j]=x     
    return res   

reslh=convolu(ksh) 
reslv=convolu(ksv) 

#method 1 for eliminating zeros
def eliminate(resl):
        minimum = min(min(resl[i]) for i in range(len(resl)))
        maximum = max(max(resl[j]) for j in range(len(resl)))
        d= maximum - minimum 
        for i in range(len(resl)):
            resl[i][:] = (((resl[i][j] - minimum)/d) for j in range(len(resl[i])))
        return resl

#method 2 for eliminating zeros
def eliminate2(resl):
    for i in range(len(resl)):
        resl[i]=[abs(j) for j in resl[i]]
    maximum = max([max(j) for j in resl])
    for i in range(len(resl)):
        resl[i][:] = [x / maximum for x in resl[i]]
    return resl
#applying zeros elimination method        
reslho = eliminate2(reslh)  
reslve = eliminate2(reslv)  
reslve = np.asarray(reslve)
for i in range(len(reslve)):
    for j in range(len(reslve[0])):
        if(reslve[i][j]>0.1):
            reslve[i][j] = 1
        else:
            reslve[i][j] = 0
reslve[:][:] = 255*reslve[:][:]
#showing the image.
cv2.namedWindow('y edge',cv2.WINDOW_NORMAL)
cv2.namedWindow('x edge',cv2.WINDOW_NORMAL)
cv2.imshow('y edge', np.asarray(reslho))
cv2.imshow('xedge', reslve)
cv2.imwrite('x edge.jpg', reslve)
cv2.waitKey(0)
cv2.destroyAllWindows()  
    

        

