#! /usr/bin/env python
import sys

from numpy import *
from scipy import *
import ols

def calcpairdist(data):
   assert data.shape[1] == 3
   ids = data[:,0] - 1
   x = data[:,1]
   y = data[:,2]
   imax = data.shape[0]
   paird = zeros(imax)
   nearestind = zeros(imax)
   nearestdist = zeros(imax)
   for i in r_[:imax]:
      for j in r_[:imax]:
         paird[j] = sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
      nearestind = row_stack([nearestind,paird.argsort(kind='merge')])
      nearestdist = row_stack([nearestdist,paird[paird.argsort(kind='merge')]])
   nearestind = nearestind[1:,1:6]
   nearestdist = nearestdist[1:,1:6]
   nearest =  column_stack([data,nearestind,nearestdist])
   sparseind = nearestdist[:,0].argsort(kind='merge')
   sparseind = sparseind[::-1]
   sparse = nearest[sparseind,:]
   return sparse

def filteroutedge(sparse, limitd=0, xmax=256, ymax=512):
   output=zeros(13)
   for i in r_[:sparse.shape[0]]:
      x=sparse[i,1]
      y=sparse[i,2]
      if (x>=limitd and x<xmax-limitd and y>=limitd and y<ymax-limitd):
         output = row_stack([output, sparse[i,:]])
   return output[1:,:]

def comparepaird(spg,spr):
   limitr = 1
   limitd = 10
   limitsqd = limitd**2
   done=repeat(False,spg.shape[0])
   outg=zeros(13)
   outr=zeros(13)   
   for i in r_[:spr.shape[0]]:
      for j in r_[:spg.shape[0]]:
         if ((not done[j]) and (abs(spg[j,8:13]-spr[i,8:13])<limitr).all() and ((spg[j,1]-spr[i,1])**2+(spg[j,2]-spr[i,2])**2<limitsqd).all()):
            outr=row_stack([outr,spr[i,:]])               
            outg=row_stack([outg,spg[j,:]])
            done[j]=True
            break
               
   savetxt("spgrn.txt", outg[1:,:], fmt='%d\t%10.5f\t%10.5f\t%d\t%d\t%d\t%d\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f')
   savetxt("spred.txt", outr[1:,:], fmt='%d\t%10.5f\t%10.5f\t%d\t%d\t%d\t%d\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f')
   return(outg[1:,:3],outr[1:,:3])

def calccoeff(grndata, reddata) :
    xGrn=grndata[:,1]
    yGrn=grndata[:,2]
    xRed=reddata[:,1]
    yRed=reddata[:,2]

    # Part I: Calculating Grn to Red transformation coefficients      
    # The function chosen is a third order polynomial function

    Grn=c_[xGrn, xGrn**2, xGrn**3, yGrn, yGrn**2, yGrn**3]

    mymodel = ols.ols(xRed,Grn,y_varnm='xRed',x_varnm=['x','x^2','x^3','y','y^2','y^3'])
    coeffGtoR_x = mymodel.b
    mymodel.summary()

    mymodel = ols.ols(yRed,Grn,y_varnm='xRed',x_varnm=['x','x^2','x^3','y','y^2','y^3'])
    coeffGtoR_y = mymodel.b
    mymodel.summary()

    coeffGtoR = hstack((coeffGtoR_x,coeffGtoR_y))

    GrnPlus=c_[repeat(1,Grn.shape[0]), Grn]
    errGtoR=sqrt((dot(GrnPlus,coeffGtoR_x)-xRed)**2+(dot(GrnPlus,coeffGtoR_y)-yRed)**2)

    # Part II: Calculating Red to Grn transformation coefficients
    # The function chosen is a third order polynomial function

    Red=c_[xRed, xRed**2, xRed**3, yRed, yRed**2, yRed**3]

    mymodel = ols.ols(xGrn, Red, y_varnm='Grn',x_varnm=['x','x^2','x^3','y','y^2','y^3'])
    coeffRtoG_x = mymodel.b
    mymodel.summary()

    mymodel = ols.ols(yGrn, Red, y_varnm='Grn',x_varnm=['x','x^2','x^3','y','y^2','y^3'])
    coeffRtoG_y = mymodel.b
    mymodel.summary()

    coeffRtoG = hstack((coeffRtoG_x,coeffRtoG_y))

    RedPlus=c_[repeat(1,Red.shape[0]), Red]
    errRtoG=sqrt((dot(RedPlus,coeffRtoG_x)-xGrn)**2+(dot(RedPlus,coeffRtoG_y)-yGrn)**2)
    
    return (coeffGtoR, coeffRtoG, errGtoR, errRtoG)

def calcErr(datain,dataout,coeff):
    xin=datain[:,1]
    yin=datain[:,2]
    xout=dataout[:,1]
    yout=dataout[:,2]

    xyin=c_[repeat(1,datain.shape[0]), xin, xin**2, xin**3, yin, yin**2, yin**3]
    err=sqrt((dot(xyin,coeff[:7])-xout)**2+(dot(xyin,coeff[7:])-yout)**2)
    return err

def main():
   """Read PolishedSpots of the alignement beads and make a table of paired distance.
   Find the five nearest beads. Sort the nearest distance in descending order so that
   beads in the sparse area will be on the top of the list. Compare the paired distance
   and x, y location in green and red tables and find the candidate pair. Then calculate
   the transform coefficients and evaluate the error. Remove the pair with which the error
   exceeds limit and re-calculate the transform coefficients"""
   errLimit = 1.0
   #   grn=genfromtxt('OriginalSpots/SpA0001.txt', skiprows=1)
   #   red=genfromtxt('OriginalSpots/SpA0002.txt', skiprows=1)
   grn=genfromtxt('PolishedSpots/PSpA0001.txt', skiprows=1)
   red=genfromtxt('PolishedSpots/PSpA0002.txt', skiprows=1)
   if red.shape[0] < 7:
      print "There are not enough beads for a unique interpolation"
      return

   # Remove the spots with which polishment failed. Keep the id and x-y location.
   grntrim=column_stack([grn[grn[:,-1]==1,0], grn[grn[:,-1]==1,4:6]])
   redtrim=column_stack([red[red[:,-1]==1,0], red[red[:,-1]==1,4:6]])

   #Calculate the paired distance
   sparseg = calcpairdist(grntrim)
   sparser = calcpairdist(redtrim)
   
   savetxt("sparsepairdgrn.txt", sparseg, fmt='%d\t%10.5f\t%10.5f\t%d\t%d\t%d\t%d\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f')
   savetxt("sparsepairdred.txt", sparser, fmt='%d\t%10.5f\t%10.5f\t%d\t%d\t%d\t%d\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\t%10.5f')

   
   grnout,redout = comparepaird(sparseg,sparser)
   cGtoR, cRtoG, eGtoR, eRtoG = calccoeff(grnout, redout)   
   coeff = hstack((cGtoR, cRtoG))
   savetxt("coefficients_ols_nonlimit.txt",coeff)

   print "Error GtoR > ",errLimit," : ",(eGtoR>errLimit).nonzero()
   print "Error RtoG > ",errLimit," : ",(eRtoG>errLimit).nonzero()   
   savetxt("alignedbeadspairNonLimit.txt", column_stack([grnout,redout,eGtoR,eRtoG]), fmt='%d\t%10.5f\t%10.5f\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f')
   indLimit = (eGtoR<=errLimit) * (eRtoG<=errLimit)
   print "indLimit==False :",(indLimit==False).nonzero()
   cGtoR, cRtoG, eGtoR, eRtoG = calccoeff(grnout[indLimit], redout[indLimit])
   coeff = hstack((cGtoR, cRtoG))
   savetxt("coefficients_ols_limit.txt",coeff)

   savetxt("alignedbeadspairLimit.txt", column_stack([grnout[indLimit],redout[indLimit],eGtoR,eRtoG]), fmt='%d\t%10.5f\t%10.5f\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f')

   
   
if __name__ == '__main__':
    main()



