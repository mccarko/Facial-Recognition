import numpy as np
from collections import namedtuple
import os
import sys
import math
os.chdir(sys.path[0])

imageType="P2"
PGMFile = namedtuple('PGMFile', ['max_shade', 'data'])

def readP2(file):
    with open(file) as f:
        lines = f.readlines()
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    assert lines[0].strip() == imageType 
    data = []
    for line in lines[1:]:
        for c in line.split():
            if(c.isdigit()):
                data.append(int(c))
    image=np.zeros((data[0],data[1]))
    x=0
    y=0
    i=3
    while i < len(data):
        image[x,y]=data[i]
        x=x+1
        if x==data[0]:
            x=0
            y=y+1
        i=i+1   
    return (PGMFile(data[2],image))

def writeP2(name,pgm):
    f=open(name,"w")
    f.write("%s \n" %imageType)
    f.write("%d %d \n" % (len(pgm.data),len(pgm.data[0])))
    f.write("%d \n" % pgm.max_shade)
    x=0
    y=0
    while y<len(pgm.data[0]):
        while x<len(pgm.data):
            f.write("%d " %pgm.data[x,y])
            x=x+1
        f.write("\n")
        x=0
        y=y+1

def brightness(pgm):
    x=0
    y=0
    totalPixels=0
    average=0
    while y<len(pgm.data[0]):
        while x<len(pgm.data):
            average+=pgm.data[x,y]        
            totalPixels+=1
            x=x+1
        x=0
        y=y+1
    return average/totalPixels/pgm.max_shade

def gaussian2D(deviation,neighbors):
  A=1/math.sqrt(2*math.pi*(deviation**2))
  kernel=np.zeros((2*neighbors+1,2*neighbors+1))
  for x in range(neighbors+1):
      for y in range(neighbors+1):
        kernel[neighbors+x,neighbors+y]=A*math.e**(-(x**2+y**2)/(2*(deviation**2)))
        kernel[neighbors+x,neighbors-y]=A*math.e**(-(x**2+y**2)/(2*(deviation**2)))
        kernel[neighbors-x,neighbors+y]=A*math.e**(-(x**2+y**2)/(2*(deviation**2)))
        kernel[neighbors-x,neighbors-y]=A*math.e**(-(x**2+y**2)/(2*(deviation**2)))
  return (kernel/np.sum(kernel))

def gaussian1D(deviation,neighbors):
  A=1/math.sqrt(2*math.pi*(deviation**2))
  kernel=np.zeros(2*neighbors+1)
  for x in range(neighbors+1):
      kernel[neighbors+x]=A*math.e**(-(x**2)/(2*(deviation**2)))
      kernel[neighbors-x]=A*math.e**(-(x**2)/(2*(deviation**2)))
  return (kernel/np.sum(kernel))

def covolve2D2D(kernel,image):
    convolution=np.zeros((len(image.data),len(image.data[0])))
    currImg=[]
    currKernel=[]
    midK=int((len(kernel)-1)/2)    
    for x in range(len(convolution)):
        for y in range(len(convolution[0])):
            edgeL=0
            edgeR=0
            edgeT=0
            edgeB=0
            if(midK>x):
                edgeL=midK-x
            if(len(image.data)-1<x+midK):
                edgeR=midK+x-(len(image.data)-1)
            if(y<midK):
                edgeT=midK-y
            if(y+midK>(len(image.data[0])-1)):
                edgeB=y+midK-(len(image.data[0])-1)
            currImg=image.data[(x-midK+edgeL):(x+midK-edgeR+1),(y-midK+edgeT):(y+midK-edgeB+1)]
            currKernel=kernel[(0+edgeL):(len(kernel)-edgeR),(0+edgeT):(len(kernel)-edgeB)]
            currKernel=currKernel/np.sum(currKernel)
            convolution[x][y]=int(np.sum(currImg*currKernel))
    return PGMFile(max_shade=image.max_shade,data=convolution)

def convolve_1d(kernel,data):
    midK=int((len(kernel)-1)/2)
    currImg=[]
    currKernel=[]
    convolution=np.zeros((len(data),len(data[0])))
    for x in range(len(convolution)):
        for y in range(len(convolution[0])):
            edgeL=0
            edgeR=0
            if(midK>x):
                edgeL=midK-x
            if(x+midK>len(data)-1):
                edgeR=x+midK-(len(data)-1)
            currImg=data[(x-midK+edgeL):(midK+x-edgeR+1),y]
            currKernel=kernel[(0+edgeL):(len(kernel)-edgeR)]
            currKernel=currKernel/np.sum(currKernel)
            convolution[x][y]=int(np.sum(currImg*currKernel/np.sum(currKernel)))
    return convolution

def convolve_1d_double(kernel,image):
    horiz=convolve_1d(kernel,image.data)
    vert=convolve_1d(kernel,np.transpose(horiz))
    return PGMFile(max_shade=image.max_shade,data=np.transpose(vert))

def detectEdge(kernel,image):
    convolution=np.zeros((len(image.data),len(image.data[0])))
    currImg=[]
    currKernel=[]
    midK=int((len(kernel)-1)/2)
    for x in range(len(convolution)):
        for y in range(len(convolution[0])):
            edgeL=0
            edgeR=0
            edgeT=0
            edgeB=0
            if(midK>x):
                edgeL=midK-x
            if(len(image.data)-1<x+midK):
                edgeR=midK+x-(len(image.data)-1)
            if(y<midK):
                edgeT=midK-y
            if(y+midK>(len(image.data[0])-1)):
                edgeB=y+midK-(len(image.data[0])-1)
            currImg=image.data[(x-midK+edgeL):(x+midK-edgeR+1),(y-midK+edgeT):(y+midK-edgeB+1)]
            currKernel=kernel[(0+edgeL):(len(kernel)-edgeR),(0+edgeT):(len(kernel)-edgeB)]
            midY=midK-edgeT
            midX=midK-edgeL
            currKernel[midX][midY]=currKernel[midX][midY]-np.sum(currKernel)
            temp=int(min(np.sum(currImg*currKernel),255))
            temp=int(max(temp,0))
            convolution[x][y]=temp
    return PGMFile(max_shade=image.max_shade,data=convolution)

def detectEdge(image):
    result=np.zeros((len(image.data),len(image.data[0])))
    xSigma=0
    ySigma=0
    newShade=0
    for x in range(len(image.data)):
        for y in range(len(image.data[0])):
            if(x==0):
                xSigma=(image.data[x+1][y]-image.data[x][y])/2
            elif(x==(len(image.data)-1)):
                xSigma=(image.data[x][y]-image.data[x-1][y])/2
            else:
                xSigma=(image.data[x+1][y]-image.data[x-1][y])/2
            
            if(y==0):
                ySigma=(image.data[x][y+1]-image.data[x][y])/2
            elif(y==(len(image.data[0])-1)):
                ySigma=(image.data[x][y]-image.data[x][y-1])/2
            else:
                ySigma=(image.data[x][y+1]-image.data[x][y-1])/2
            change=math.sqrt((xSigma**2)+(ySigma**2))
            result[x][y]=change
            if(change>newShade):
                newShade=change
    return PGMFile(max_shade=newShade,data=result)

def supressEdge(image):
    result=np.zeros((len(image.data),len(image.data[0])))
    xSigma=0
    ySigma=0
    for x in range(len(image.data)):
        for y in range(len(image.data[0])):
            if(x==0):
                xSigma=(image.data[x+1][y]-image.data[x][y])/2
            elif(x==(len(image.data)-1)):
                xSigma=(image.data[x][y]-image.data[x-1][y])/2
            else:
                xSigma=(image.data[x+1][y]-image.data[x-1][y])/2
            
            if(y==0):
                ySigma=(image.data[x][y+1]-image.data[x][y])/2
            elif(y==(len(image.data[0])-1)):
                ySigma=(image.data[x][y]-image.data[x][y-1])/2
            else:
                ySigma=(image.data[x][y+1]-image.data[x][y-1])/2
            result[x][y]=image.data[x][y]

            theta=math.atan2(abs(ySigma),abs(xSigma))
            
            if(theta<=math.pi/8 or theta>7*math.pi/8):
                if(0<x):
                    if(image.data[x][y]<image.data[x-1][y]):
                        result[x][y]=0
                if(x<len(image.data)-1):
                    if(image.data[x][y]<image.data[x+1][y]):
                        result[x][y]=0
            elif(theta<=3*math.pi/8):
                if(0<x and y<len(image.data[0])-1):
                    if(image.data[x][y]<image.data[x-1][y+1]):
                        result[x][y]=0
                if(x<len(image.data)-1 and 0<y):
                    if(image.data[x][y]<image.data[x+1][y-1]):
                        result[x][y]=0
            elif(theta<=5*math.pi/8):
                if(0<y):
                    if(image.data[x][y]<image.data[x][y-1]):
                        result[x][y]=0
                if(y<len(image.data[0])-1):
                    if(image.data[x][y]<image.data[x][y-1]):
                        result[x][y]=0
            else:
                if(0<x and 0<y):
                    if(image.data[x][y]<image.data[x-1][y-1]):
                        result[x][y]=0
                if(x<len(image.data)-1 and y<len(image.data[0])-1):
                    if(image.data[x][y]<image.data[x+1][y+1]):
                        result[x][y]=0
    return PGMFile(max_shade=image.max_shade,data=result)

def supressNoise(image,low,high):
    result=np.zeros((len(image.data),len(image.data[0])))
    for x in range(len(image.data)):
        for y in range(len(image.data[0])):
            result[x][y]=image.data[x][y]
            ignore=False
            if(image.data[x][y]<low):
                ignore=True
            elif(image.data[x][y]<high):
                if(0<y):
                    if(0<x and not ignore):
                        ignore=image.data[x-1][y-1]<high
                    if(x<len(image.data)-1 and not ignore):
                        ignore=image.data[x+1][y-1]<high
                    if(not ignore):
                        ignore=image.data[x][y-1]<high
                if(0<x and not ignore):
                    ignore=image.data[x-1][y]<high
                if(x<len(image.data)-1 and not ignore):
                    ignore=image.data[x+1][y]<high
                if(y<len(image.data[0]-1) and not ignore):
                    if(0<x and not ignore):
                        ignore=image.data[x-1][y+1]<high
                    if(x<len(image.data)-1 and not ignore):
                        ignore=image.data[x+1][y+1]<high
                    if(not ignore):
                        ignore=image.data[x][y+1]<high
            if(ignore):
                result[x][y]=0
    return PGMFile(max_shade=image.max_shade,data=result)