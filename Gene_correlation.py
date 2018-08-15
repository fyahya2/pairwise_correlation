import numpy as np
import pandas as pd
import scipy as sc
import math
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeSVD, BiScaler, KNN, NuclearNormMinimization, SoftImpute


#Computing significances of subnetworks

def fraction(eigenexprssionsi,M):
 #   ai_fractions = ai_sSquare / np.sum(ai_sSquare)
    #print(a1_fractions)
    lambdaSum = np.sum(eigenexprssionsi)
    aiFrac = np.zeros(M)
    for i in range(0, M):
        aiFrac[i] = eigenexprssionsi[i] / lambdaSum
    return aiFrac
#---------------------------------------------

#Calculating entropy

def ent(data):
    entropy = 0
    for i in range(0,len(data)):
        entropy = entropy + (data[i] * math.log2(data[i]))
            
    entropy = -1/math.log2(12) *entropy
    return entropy

#---------------------------------------------

#Writing on the file

def Clear_Cell_Cycle(line, outfile):
    outfile.write("\t".join(line.split()[7:25]) + "\n")
    return
def Clear_Bind(line, outfile):
    outfile.write("\t".join(line.split()[7:19]) + "\n")
    return

#---------------------------------------------

#Computing HOEVD for each individual network

def HOEVD(ai, ai_sSquare, eigenarrays, M, d):
    ai_hoevd = np.zeros((d,d));
    
    #Independent subnetworks
    
    for m in range(0,5):
        ai_hoevdTemp = ai_sSquare[m] * (np.outer(eigenarrays[:,m], eigenarrays[:,m]))
        ai_hoevd = ai_hoevd + ai_hoevdTemp

    subtracted_ai = ai- ai_hoevd

    #Make this matrix rectangle
    
    index = d*d
    k=0
    rec_a = np.zeros((index,1));
    for i in range(0,d):
        for j in range(0,d):
            rec_a[k][0] = subtracted_ai[i][j]
            k = k+1

    #Rank 2 couplings
            
    i=0
    matrix = [[[0 for i in range(d)] for j in range(d)] for k in range(10)]
    f = open("matrix.txt", 'w')
    for m in range(0,4):
        for l in range(m+1, 4):
            matrix[i] = np.outer(eigenarrays[:,l], eigenarrays[:,m]) + np.outer(eigenarrays[:,m], eigenarrays[:,l])
            f.write("%s" % matrix[i])
            i = i+1

    #Finding correlations for couplings
            
    coef = np.zeros((index,10));
    k=0
    for m in range(0,d):
        for n in range(0,d):
            for l in range(0,9):
                coef[k][l] = matrix[l][m][n]
            k=k+1

    x = np.linalg.lstsq(coef, rec_a)[0]
    print("corelations for couplings")
    print(x)
    print('\n')
    return

#--------------------------------------------------

#Finding relevant data of genes in datasets

def relData(interSec, sig, genes, M):

    b1_e1 = np.zeros((1588,M))
    k=0
    
    for j in range(0,len(genes)):
        if genes[j] in interSec:
            b1_e1[k] = sig[j,:]
            k=k+1           
    newData = b1_e1
    
    print("Dimension of intersectinon matrix of e1 and b1 of b1 ")
    print(newData.shape)
    print('\n')
    return newData

#------------------------------------------------

def main():

    #Enter the raw data file (main signal matrix e1)
    
    with open('Cell_Cycle_Expresion.txt','r') as infile:
        with open("Clear_Cell_Cycle.txt", "w") as outfile:
            for line in infile:
                Clear_Cell_Cycle(line, outfile);

    #Removing first row of the data file
                
    with open("Clear_Cell_Cycle.txt", 'r') as fin:
        data = fin.read().splitlines(True)
    with open("Clear_Cell_Cycle.txt", 'w') as fout:
        fout.writelines(data[1:])

    #Keeping name of the genes and features of e1 (Cell_Cycle_Expresion)
        
    with open('Cell_Cycle_Expresion.txt', "r") as infile:
        with open("GeneNames_Cell_Cycle.txt", "w") as outfile:
            for line in infile:
                outfile.write("\t".join(line.split()[0]) + "\n")
                
    with open("GeneNames_Cell_Cycle.txt", 'r') as fin:
        data = fin.read().splitlines(True)
    with open("GeneNames_Cell_Cycle.txt", 'w') as fout:
        fout.writelines(data[1:])

    #Creating list of signal genes
        
    f = open("GeneNames_Cell_Cycle.txt",'r')
    sig_genes = f.readlines()
    print("Length of signal genes of e1: ")
    print(len(sig_genes))
    print('\n')
               
    #Changing NULL to 0
        
    with open("Clear_Cell_Cycle.txt", 'r') as fin2:
        data = fin2.read().splitlines(True)
    with open("Clear_Cell_Cycle.txt", 'w') as fout2:
        for line in data:
            fout2.write(line.replace('Null', '0'))

    #Creating matrix e1
            
    sig_matrix = np.loadtxt('Clear_Cell_Cycle.txt')
    print("Dimension of raw e1: ")
    print(sig_matrix.shape)
    print('\n')

#---------------------------------------------------
    
    #Filling missing data in e1
    
    sig_matrix[sig_matrix == 0] = np.NaN
    X_incomplete = sig_matrix
    #imputer = Imputer()
    #transformed_sig_matrix = imputer.fit_transform(sig_matrix)
    #Count the number of NaN values in each column
    #print(np.isnan(transformed_sig_matrix).sum())
    #sig_matrix = transformed_sig_matrix

    # Use SVD
    X_filled = IterativeSVD().complete(X_incomplete)
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    X_filled_knn = KNN(k=5).complete(X_incomplete)
 #   svd_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
 #   print("IterativeSVD MSE: %f" % svd_mse)
    
    # matrix completion using convex optimization to find low-rank solution
    # that still matches observed values. Slow!
    #X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)

    # print mean squared error for the three imputation methods above
    #nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()
    #print("Nuclear norm minimization MSE: %f" % nnm_mse)

 #   knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
 #   print("knnImpute MSE: %f" % knn_mse)

    sig_matrix = X_filled_knn

    #Center the expressions of genes
    sig_npArray = np.array(sig_matrix)
    sig_mean = np.mean(sig_npArray, axis=0)
    print("Mean of e1 at it's time level:")
    print(sig_mean)
    for i in range(0, 18):
        sig_matrix[:,i] = sig_matrix[:, i] - sig_mean[i]
    print("\n")

    
#--------------------------------------------------
    
    #Calculating svd of e1   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays1, eigenexpressions1, eigengenes1 = np.linalg.svd(sig_matrix, full_matrices=False)

    #Creating a1 by e1
    
    a1 = np.dot(sig_matrix,sig_matrix.transpose())
    print("Dimension of raw a1: ")
    print(a1.shape)
    print('\n')
    
    #Calculating Evd of a1:network1
    
    eigenarrays1_trans = eigenarrays1.transpose()
    a1_sSquare = np.square(eigenexpressions1)

    #Calculating significance of subnetworks of a1

    M1=4;
    a1Frac = fraction(eigenexpressions1,M1)
    print("Expression correlations for 4 most significant subnetworks of a1(%)")
    print(a1Frac*100)
    print('\n')
    a1_entropy = ent(a1Frac)
    print("Entropy of matrix a1")
    print(a1_entropy)
    print('\n')

#---------------------------------------------------   
    #Enter the raw data files for basic signals b1, b2, and b3 
   
    #b1
    with open('Cell_Cycle_Binding.txt') as infile2:
        with open("Clear_Cycle_Bin.txt", "w") as outfile2:
            for line in infile2:
                Clear_Bind(line, outfile2);

    #Removing first row of the data file
                
    with open("Clear_Cycle_Bin.txt", 'r') as fin2:
        data = fin2.read().splitlines(True)
    with open("Clear_Cycle_Bin.txt", 'w') as fout2:
        fout2.writelines(data[1:])

    #Creating matrix basis signal b1
        
    sig_basis1 = np.loadtxt('Clear_Cycle_Bin.txt')
    print("Dimension of b1(Cell Cycle Binding): ")
    print(sig_basis1.shape)
    print('\n')

    #Keeping name of the genes and features of b1 (Cell_Cycle_Binding)
        
    with open('Cell_Cycle_Binding.txt') as infile:
        with open("GeneNames_Cycle_Bin.txt", "w") as outfile:
            for line in infile:
                outfile.write("\t".join(line.split()[0]) + "\n")
                
    with open("GeneNames_Cycle_Bin.txt", 'r') as fin:
        data = fin.read().splitlines(True)
    with open("GeneNames_Cycle_Bin.txt", 'w') as fout:
        fout.writelines(data[1:])

    #Creating list of signal genes of b1
        
    f = open("GeneNames_Cycle_Bin.txt",'r')
    b1_genes = f.readlines()
    print("Length of signal genes of b1: ")
    print(len(b1_genes))
    print('\n')

    #Find intersection of e1 and b1
    
    interSec = set(sig_genes).intersection(b1_genes)
    print("Length of intersection of e1 and b1")
    print(len(interSec))
    print('\n')

    #Finding relevant data of intersection(e1, b1) in e1
    M=18
    sig_matrix = relData(interSec, sig_matrix, sig_genes, M)

    #Finding relevant data of intersection(e1, b1) in b1
    M=12
    sig_basis1 = relData(interSec, sig_basis1, b1_genes, M)
    
    #Devide signal matrix by mean to convert signals to DNA binding
    
    sig_npArray = np.array(sig_basis1)
    basis1_mean = np.mean(sig_npArray, axis=1)
    print("Mean of b1 for gean measurments:")
    print(basis1_mean)
    for i in range(0, 1588):
        sig_basis1[i, :] = sig_basis1[i, :] / basis1_mean[i]
    print("\n")

    #Calculating svd of b1   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays2, eigenexprssions2, eigengenes2 = np.linalg.svd(sig_basis1, full_matrices=False)
    print("Eigenexpresions of partial b1")
    print(eigenexprssions2)
    print('\n')

    #Computing entropy of b1

    M1=12
    b1Frac = fraction(eigenexprssions2,M1)
    print("Expression correlations for most significant subnetworks of b1(%)")
    print(a1Frac*100)
    print('\n')
    b1_entropy = ent(b1Frac)
    #print(b1Frac[1])
    print("Entropy of partial b1")
    print(b1_entropy)
    print('\n')
    
    #b2
    
    with open('Develop_Binding.txt') as infile3:
        with open("Clear_Dev_Bin.txt", "w") as outfile3:
            for line in infile3:
                Clear_Bind(line, outfile3);

    #Removing first row of the data file
                
    with open("Clear_Dev_Bin.txt", 'r') as fin3:
        data = fin3.read().splitlines(True)
    with open("Clear_Dev_Bin.txt", 'w') as fout3:
        fout3.writelines(data[1:])

    #Creating matrix basis signal b2
        
    sig_basis2 = np.loadtxt('Clear_Dev_Bin.txt')
    print("Dimension of b2: ")
    print(sig_basis2.shape)
    print('\n')
    
    #b3
    
    with open('Biosynthesis_Binding.txt') as infile4:
        with open("Clear_Biosyn_Bin.txt", "w") as outfile4:
            for line in infile4:
                Clear_Bind(line, outfile4);

    #Removing first row of the data file
                
    with open("Clear_Biosyn_Bin.txt", 'r') as fin4:
        data = fin4.read().splitlines(True)
    with open("Clear_Biosyn_Bin.txt", 'w') as fout4:
        fout4.writelines(data[1:])

    #Creating matrix basis signal b3
        
    sig_basis3 = np.loadtxt('Clear_Biosyn_Bin.txt')
    print("Dimension of b3: ")
    print(sig_basis3.shape)
    print('\n')
#----------------------------------------------------
    
    #pseudoInverse projection to create a2, a3, and a4
    #a2
    b1_pseoudoInv = np.linalg.pinv(sig_basis1)
    project1 = np.dot(sig_basis1, b1_pseoudoInv)
    sig2_matrix = np.dot(project1, sig_matrix)
    print("Dimension of e2: ")
    print(sig2_matrix.shape)
    print('\n')
   
    #Calculating svd of e2   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays2, eigenexpressions2, eigengenes2 = np.linalg.svd(sig2_matrix, full_matrices=False)
    print("Eigenexpresions of e2")
    print(eigenexpressions2)
    print("\n")
    
    #Creating a2 by e2
    
    a2 = np.dot(sig2_matrix,sig2_matrix.transpose())
    
    #Calculating Evd of a2:network2
    
    eigenarrays2_trans = eigenarrays2.transpose()
    a2_sSquare = np.square(eigenexpressions2)

    #Calculating significance of subnetworks of a2

    M2=2;
    a2Frac = fraction(eigenexpressions2,M2)
    print("Expression correlations for most significant subnetworks of a2(%)")
    print(a2Frac*100)
    print('\n')
    a2_entropy = ent(a2Frac)
    print("Entropy of matrix a2(should be .49)")
    print(a1_entropy)
    print("fraction of first eigenvalue of a2")
    print(a2Frac[0])
    print('\n')

    #a3
    b2_pseoudoInv = np.linalg.pinv(sig_basis2)
    project2 = np.dot(sig_basis1, b1_pseoudoInv)
    sig3_matrix = np.dot(project2, sig_matrix[0:2120])
    
    #Picking 2120 genes out of the matrix
    
    sig3_matrix = sig3_matrix[0:2120,:]
    print("dimension of e3: ")
    print(sig3_matrix.shape)
    print('\n')
    
    #Calculating svd of e3   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays3, eigenexprssions3, eigengenes3 = np.linalg.svd(sig3_matrix, full_matrices=False)

    #Creating a3 by e3
    
    a3 = np.dot(sig_matrix,sig_matrix.transpose())

    #Calculating Evd of a3:network3
    
    eigenarrays3_trans = eigenarrays3.transpose()
    a3_sSquare = np.square(eigenexprssions3)
    

    #Calculating significance of subnetworks of a3
    
    a3_fractions = a3_sSquare / np.sum(a3_sSquare)
    

    #a4
    
    b3_pseoudoInv = np.linalg.pinv(sig_basis3)
    project3 = np.dot(sig_basis1, b1_pseoudoInv)
    sig4_matrix = np.dot(project3, sig_matrix[0:2120])

    #Picking 2120 genes out of the matrix
    
    sig4_matrix = sig4_matrix[0:2120,:]
    print("dimension of e4: ")
    print(sig4_matrix.shape)    
    print('\n')
    #Calculating svd of e4 = UsV   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays4, eigenexprssions4, eigengenes4 = np.linalg.svd(sig4_matrix, full_matrices=False)
    
    #Creating a4 by e4
    
    a4 = np.dot(sig_matrix,sig_matrix.transpose())
    
    #Calculating Evd of a4:network4
    
    eigenarrays4_trans = eigenarrays4.transpose()
    a4_sSquare = np.square(eigenexprssions4)

    #Calculating significance of subnetworks of a4
    
    a4_fractions = a4_sSquare / np.sum(a4_sSquare)

#-------------------------------------------------
    
    #Tensor Decomposition

    #Picking 1588 genes out of the matrix
    
 #   a_T = a1 + a2 + a3 + a4
    #print(a_T.shape)
    
    #Appending signal matrices e1 to e4
    
    sig_appendTemp = np.concatenate((sig_matrix.transpose(), sig2_matrix.transpose(), sig3_matrix.transpose(), sig4_matrix.transpose()), axis=0)
    sig_append = sig_appendTemp.transpose()
    print("dimension of appended e")
    print(sig_append.shape)
    print('\n')
    
    #Calculating svd of appended e = UsV   U: eigenarray, s: eigenexpresions, V: eigengenes
    
    eigenarrays, eigenexprssions, eigengenes = np.linalg.svd(sig_append, full_matrices=True)

    #Calculating HOEVD of overall network a_T
    
    a_T_sSquare = np.square(eigenexprssions)
    a_T_append = np.dot(sig_append,sig_append.transpose())
    #print(a_T_sSquare.shape)

    #HOEVD for individual networks
    M = 18 + 12 + 12 + 8
    M_couple = M *( M-1)/2

    M1 = 15
    d = 1588
    
    #Picking 1588 genes out of the matrix
    
    a1 = a1[0:1588,0:1588]
    
    HOEVD(a1, a1_sSquare, eigenarrays, M1, d)
    
    M2 = 12
    d = 1588
    HOEVD(a2, a2_sSquare, eigenarrays, M2, d)


    

if __name__ == '__main__':
    main()
            
    
