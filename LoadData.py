from __future__ import print_function
import sys
import os
import numpy as np
import scipy.io as sio
import glob
import random

def import_files():

    flair_file_list = glob.glob("F:\Internship2016\Data\UpdatedTransformedRegionsOfInterest\Flair" + "/*.mat")
    tmax_file_list = glob.glob("F:\Internship2016\Data\UpdatedTransformedRegionsOfInterest\Tmax" + "/*.mat")
    ttp_file_list = glob.glob("F:\Internship2016\Data\UpdatedTransformedRegionsOfInterest\TTP" + "/*.mat")
    adc_file_list = glob.glob("F:\Internship2016\Data\UpdatedTransformedRegionsOfInterest\Adc" + "/*.mat")


    #all_bad_examples=np.load("F:\\Internship2016\\Data\\Lists\\bad_examples.npy")
    all_bad_examples=[0, 9, 12, 13, 16, 17, 18, 22, 28, 29, 32, 34, 37, 47, 50, 58, 60, 65, 69, 70, 75, 76, 78, 80, 84, 89, 92, 95, 104, 110, 116, 122, 123, 124, 128, 130, 136, 139, 147, 151, 159, 160, 166, 170, 171, 179, 182, 183, 186, 195, 196, 200, 201, 202, 208, 210, 217, 221, 224, 225, 228, 232, 234, 239]

    flairIncluded=np.zeros((241,220,172))
    tmaxIncluded=np.zeros((241,1,220,172))
    ttpIncluded=np.zeros((241,1,220,172))
    adcIncluded=np.zeros((241,1,220,172))
    adcRoiIncluded=np.zeros((241,1,220,172))
    counter=0
    for i in range(len(flair_file_list)):
        flair_file=flair_file_list[i]
        tmax_file=tmax_file_list[i]
        ttp_file=ttp_file_list[i]
        adc_file=adc_file_list[i]

        flair_file=sio.loadmat(flair_file)
        tmax_file=sio.loadmat(tmax_file)
        ttp_file=sio.loadmat(ttp_file)
        adc_file=sio.loadmat(adc_file)
        if i not in all_bad_examples:
            slc=np.array(flair_file['refFlair'])
            a1,a2,a3=np.array(flair_file['roiTransformedFlair']).shape
            if(slc>a3):
                slc=a3
            a=np.array(flair_file['roiTransformedFlair'])[:, :, (slc-1)]
            if a.sum()>400:
                a=a.reshape(1,220,172)
                flairIncluded[counter]=a
        #TMAX
                slc=np.array(tmax_file['refPwi'])
                a1,a2,a3=np.array(tmax_file['transformedPwi']).shape
                if(slc>a3):
                    slc=a3
                a=np.array(tmax_file['transformedPwi'])[:, :, (slc-1)]
                a=a.reshape(1,1,220,172)
                #a=a-a[0,0,10,10]
        #         low_values_indices = a < (a[0,0,10,10]-2)  # Where values are low
        #         high_values_indices = a > 80
        #         inside_values_indices = a > 0
        #         a[low_values_indices] = a.mean()
        #         a[high_values_indices] = a.mean()
        #         a[inside_values_indices] /= 5
                tmaxIncluded[counter]=a
        #TTP
                slc=np.array(ttp_file['refPwi'])
                a1,a2,a3=np.array(ttp_file['transformedPwi']).shape
                if(slc>a3):
                    slc=a3
                a=np.array(ttp_file['transformedPwi'])[:, :, (slc-1)]
                a=a.reshape(1,1,220,172)
                #high_values_indices = a > 70
                #a[high_values_indices] = a.mean()+10
                #a = a-a[0,0,10,10]
                #inside_indices = a > 0.1
                #a[inside_indices]-= 20# a.mean()
                #low_values_indices = a < -1
                #a[low_values_indices] = a.mean()
        #         high_values_indices = a > 80
        #         a[high_values_indices] = a.mean()+10
                #print(a.mean(), a.max(),a.min())
                ttpIncluded[counter] = a
        #ADC
                slc=np.array(adc_file['refADC'])
                a1,a2,a3=np.array(adc_file['transformedAdc']).shape
                if(slc>a3):
                    slc=a3
                a=np.array(adc_file['transformedAdc'])[:, :, (slc-1)]
                a=a.reshape(1,1,220,172)/200
                adcIncluded[counter]=a
                #slc=np.array(adc_file['refADC'])
        #         a1,a2,a3=np.array(adc_file['roiTransformedAdc']).shape
        #         if(slc>a3):
        #             slc=a3
        #         a=np.array(adc_file['roiTransformedAdc'])[:, :, (slc-1)]
        #         a=a.reshape(1,1,220,172)
        #         adcRoiIncluded[counter]=a
                #print(counter)
                counter+=1

    flairIncluded=flairIncluded[:counter]
    tmaxIncluded=tmaxIncluded[:counter]
    ttpIncluded=ttpIncluded[:counter]
    adcIncluded=adcIncluded[:counter]

    return(flairIncluded,tmaxIncluded,ttpIncluded,adcIncluded)


def extract_points (flairIncluded, tmaxIncluded, ttpIncluded, adcIncluded, examplesIncluded, ratio, minFixed):
    examples, rows, columns = flairIncluded.shape

    lesionPoints=np.zeros((0,4))
    nonLesionPoints=np.zeros((0,4))
    for e in range(examplesIncluded):
        temp=np.zeros((0,4))
        nontemp=np.zeros((0,4))
        num_lesionPoints = int(np.sum(flairIncluded[e,:,:]))
        for i in range(25, rows-25):
            for j in range(25, columns-25):
                if (ttpIncluded[e,0,i,j]>0):
                    if (flairIncluded[e, i, j] == 1):
                        temp=np.concatenate((temp,np.array([[e, i, j, 1]])))
                    elif(flairIncluded[e, i, j] == 0):
                        a=np.array([e,i,j,1]).reshape(1,4)
                        nontemp=np.concatenate((nontemp,np.array([[e, i, j, 0]])))
        temp = random.sample(temp, min(int(num_lesionPoints*ratio), minFixed))
        nontemp = random.sample(nontemp, min(int(num_lesionPoints*ratio), minFixed))
        temp=np.array(temp)
        lesionPoints=np.array(lesionPoints)
        if(e==0):
            lesionPoints=np.array(temp)
            nonLesionPoints=np.array(nontemp)
        else:
            lesionPoints=np.concatenate((lesionPoints,temp))
            nonLesionPoints=np.concatenate((nonLesionPoints,nontemp))


    allPoints=np.concatenate((lesionPoints,nonLesionPoints))
    totalPoints, unused = allPoints.shape

    return(allPoints, totalPoints)

def extract_patches(allPoints, totalPoints, patchSize1, patchSize2):

    # patchSize1=25
    # patchSize2=49
    smallInputsTmax=np.zeros((totalPoints,1,patchSize1,patchSize1))
    smallInputsTTP=np.zeros((totalPoints,1,patchSize1,patchSize1))
    smallInputsADC=np.zeros((totalPoints,1,patchSize1,patchSize1))
    # smallInputsADCroi=np.zeros((totalPoints,1,patchSize1,patchSize1))
    largeInputsTmax=np.zeros((totalPoints,1,patchSize2,patchSize2))
    largeInputsTTP=np.zeros((totalPoints,1,patchSize2,patchSize2))
    largeInputsADC=np.zeros((totalPoints,1,patchSize2,patchSize2))
    # largeInputsADCroi=np.zeros((totalPoints,1,patchSize2,patchSize2))
    outputsFlair=np.zeros((totalPoints))

    for i in range(0,totalPoints):
        e,r,c,l = allPoints[i]
        smallInputsTmax[i,0] = tmaxIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        smallInputsTTP[i,0] = ttpIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        smallInputsADC[i,0] = adcIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        # smallInputsADCroi[i,0] = adcRoiIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        outputsFlair[i]=l
        largeInputsTmax[i,0] = tmaxIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        largeInputsTTP[i,0] = ttpIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        largeInputsADC[i,0] = adcIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        # largeInputsADCroi[i,0] = adcRoiIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]

    trainFlair = np.zeros((totalPoints, 1, 49, 49))
    for i in range(totalPoints):
        trainFlair[i, 0, 0, 0] = outputsFlair[i]

    breakPoint = int(0.8 * totalPoints)

    newLargeTrainTmax = largeInputsTmax[:breakPoint]
    newLargeTrainTTP = largeInputsTTP[:breakPoint]
    newLargeTrainADC = largeInputsADC[:breakPoint]
    # newLargeTrainADCroi = largeInputsADCroi[:breakPoint]
    newTrainFlair = trainFlair[:breakPoint]

    newLargeValTmax = largeInputsTmax[breakPoint:]
    newLargeValTTP = largeInputsTTP[breakPoint:]
    newLargeValADC = largeInputsADC[breakPoint:]
    # newLargeValADCroi = largeInputsADCroi[breakPoint:]
    newValFlair = trainFlair[breakPoint:]


    return (newLargeTrainTmax, newLargeTrainTTP, newLargeTrainADC, newTrainFlair, newLargeValTmax, newLargeValTTP, newLargeValADC, newValFlair)


def rot_Data()