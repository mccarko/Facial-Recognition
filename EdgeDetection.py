import numpy as np
from collections import namedtuple
import os
import sys
os.chdir(sys.path[0])
import imgLibrary


if(len(sys.argv)>1):
    name=sys.argv[1]
    image=imgLibrary.readP2(name)
    smooth=imgLibrary.covolve2D2D(imgLibrary.gaussian2D(3,4),image)
    edges=imgLibrary.detectEdge(smooth)
    thin=imgLibrary.supressEdge(edges)
    final=imgLibrary.supressNoise(thin,0.06*thin.max_shade,0.16*thin.max_shade)
    imgLibrary.writeP2("smooth.pgm",smooth)
    imgLibrary.writeP2("edges.pgm",edges)
    imgLibrary.writeP2("thin.pgm",thin)
    imgLibrary.writeP2("final.pgm",final)
