import characterization
import numpy as np
from skimage import filters, util


def MergeWatershed(src1,src2):
    
    # src1: label image/matrix
    # src2: original image
    
    temp1 = np.copy(src1)
    temp2 = np.copy(src2)
    Mat = []
    properties = characterization.properties(temp1)
    #mean_area = np.mean(properties[0])
    #mean_solidity = np.mean(properties[1])
    #matches_area = [matches for matches in properties[0] if matches >= mean_area]
    #matches_solidity = [matches for matches in properties[1] if matches <= mean_solidity]
    matches_index_area = [i for i,x in enumerate(properties[0]) if x >= 100]
    matches_index_solidity = [i for i,x in enumerate(properties[1]) if x <= 0.9]
    for i in range (0,len(matches_index_area)):
        for j in range (0,len(matches_index_solidity)):
            if matches_index_area[i] == matches_index_solidity[j]:
                Mat.append(matches_index_area[i])
    
    temp2[temp1!=0] = 255    

    
    Thresholded = filters.threshold_otsu(temp2)
    BT_temp2 = util.invert(temp2 <= Thresholded)
    
    return Mat, BT_temp2

def isMember(A,B):
    # A : label image/matrix
    # B : list
    
    temp = np.copy(A)
    for x in range(0,A.shape[0]):
        for y in range(0,A.shape[1]):
            temp[x][y] = isFound(A[x][y],B)

    return temp

def isFound(A, B):
    if(A in B ):
        return True
    else: return False

def changeRL(A,B):
    # A : watershed matrix
    # B : watershed riged lines
    tempA = np.copy(A)
    tempB = np.copy(B)
    
    for x in range(0,tempB.shape[0]):
        for y in range(0,tempB.shape[1]):
            if (tempB[x][y] == True):
                tempA[x][y] = 0

    return tempA

def change(A,B):
    # A : matrix Distance
    # B : boolean matrix
    
    tempA = np.copy(A).astype(np.float)
    tempB = np.copy(B)
    
    for x in range(0,tempA.shape[0]):
        for y in range(0,tempA.shape[1]):
            if (tempB[x][y] == False):
                tempA[x][y] = -float('Inf')
    return tempA
    
def change2(src1,src2):
    temp1 = np.copy(src1)
    temp2 = np.copy(src2)
    
    temp2[temp1!=0] = 255 
    
    return temp2

