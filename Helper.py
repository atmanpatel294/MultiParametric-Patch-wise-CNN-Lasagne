#This file contains the helper functions that import files, extract patches, rotate them to generate more samples.

from __future__ import print_function
import sys
import os
import numpy as np
import scipy.io as sio
import glob
import random

def import_files():

    file_list = glob.glob(path_to_the_folder_containing_all_files)
    
    imageContainer = np.zeros((no_of_images, height, width))
    
    counter=0
    
    for i in range(len(file_list)):
        file_path=file_list[i]

        load_file=sio.loadmat(file_path) #if it is a .mat file
        #otherwise use functions corresponding to your file format

        a1,a2,a3=np.array(load_file).shape
        
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

            a=a-a[0,0,10,10]
            low_values_indices = a < -3  # Where values are low
            high_values_indices = a > 30
            inside_values_indices = a > 1
            a[low_values_indices] = a[inside_values_indices].mean()
            a[high_values_indices] = 6
            a[inside_values_indices] /= 1.5
            a=np.array(a, int)

            tmaxIncluded[counter]=a
    #TTP
            slc=np.array(ttp_file['refPwi'])
            a1,a2,a3=np.array(ttp_file['transformedPwi']).shape
            if(slc>a3):
                slc=a3
            a=np.array(ttp_file['transformedPwi'])[:, :, (slc-1)]
            a=a.reshape(1,1,220,172)
            a=a-a[0,0,10,10]
            low_values_indices = a < -3  # Where values are low
            high_values_indices = a > 80
            inside_values_indices = a > 1
            a[low_values_indices] = a[inside_values_indices].mean()
            a[high_values_indices] = 50
            a[inside_values_indices] /= 4
            a=np.array(a, int)
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


def extract_points (flairIncluded, tmaxIncluded, ttpIncluded, adcIncluded, exStart, exEnd, ratioLesion, maxLesion, ratioNonLesion, maxNonLesion):
    examples, rows, columns = flairIncluded.shape

    lesionPoints=np.zeros((0,4))
    nonLesionPoints=np.zeros((0,4))
    for e in range(exStart, exEnd):
        temp=np.zeros((0,4))
        nontemp=np.zeros((0,4))
        num_lesionPoints = int(np.sum(flairIncluded[e,:,:]))
        for i in range(25, rows-25):
            for j in range(25, columns-25):
                if (ttpIncluded[e,0,i,j]>0):
                    if (flairIncluded[e, i, j] == 1):
                        temp=np.concatenate((temp,np.array([[e, i, j, 1]])))
                    elif(flairIncluded[e, i, j] == 0):
                        # a=np.array([e,i,j,1]).reshape(1,4)
                        nontemp=np.concatenate((nontemp,np.array([[e, i, j, 0]])))
        temp = random.sample(temp, min(int(num_lesionPoints*ratioLesion), maxLesion))
        nontemp = random.sample(nontemp, min(int(num_lesionPoints*ratioNonLesion), maxNonLesion))
        temp=np.array(temp)
        lesionPoints=np.array(lesionPoints)
        if(e==0):
            lesionPoints=np.array(temp)
            nonLesionPoints=np.array(nontemp)
        else:
            lesionPoints=np.concatenate((lesionPoints,temp))
            nonLesionPoints=np.concatenate((nonLesionPoints,nontemp))

    allPoints=np.concatenate((lesionPoints,nonLesionPoints))
    np.random.shuffle(allPoints)
    totalPoints, unused = allPoints.shape
    numLesionPoints, unused = lesionPoints.shape
    numNonLesionPoints, unused = nonLesionPoints.shape
    return(allPoints, totalPoints, numLesionPoints, numNonLesionPoints)


def extract_patches(allPoints, totalPoints, tmaxIncluded, ttpIncluded, adcIncluded, patchSize1, patchSize2):

    # patchSize1=25
    # patchSize2=49
    smallInputsTmax=np.zeros((totalPoints,1,patchSize2,patchSize2))
    smallInputsTTP=np.zeros((totalPoints,1,patchSize2,patchSize2))
    smallInputsADC=np.zeros((totalPoints,1,patchSize2,patchSize2))
    # smallInputsADCroi=np.zeros((totalPoints,1,patchSize1,patchSize1))
    largeInputsTmax=np.zeros((totalPoints,1,patchSize1,patchSize1))
    largeInputsTTP=np.zeros((totalPoints,1,patchSize1,patchSize1))
    largeInputsADC=np.zeros((totalPoints,1,patchSize1,patchSize1))
    # largeInputsADCroi=np.zeros((totalPoints,1,patchSize2,patchSize2))
    outputsFlair=np.zeros((totalPoints))

    for i in range(0,totalPoints):
        e,r,c,l = allPoints[i]
        smallInputsTmax[i,0] = tmaxIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        smallInputsTTP[i,0] = ttpIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        smallInputsADC[i,0] = adcIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        # smallInputsADCroi[i,0] = adcRoiIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        outputsFlair[i]=l
        largeInputsTmax[i,0] = tmaxIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        largeInputsTTP[i,0] = ttpIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        largeInputsADC[i,0] = adcIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
        # largeInputsADCroi[i,0] = adcRoiIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]

    # trainFlair = np.zeros((totalPoints, 1, 49, 49))
    # for i in range(totalPoints):
    #     trainFlair[i, 0, 0, 0] = outputsFlair[i]

    breakPoint = int(totalPoints)

    largeInputsTmax = largeInputsTmax[:breakPoint]
    largeInputsTTP = largeInputsTTP[:breakPoint]
    largeInputsADC = largeInputsADC[:breakPoint]
    # newLargeTrainADCroi = largeInputsADCroi[:breakPoint]
    outputsFlair = outputsFlair[:breakPoint]

    smallInputsTmax = smallInputsTmax[:breakPoint]
    smallInputsTTP = smallInputsTTP[:breakPoint]
    smallInputsADC = smallInputsADC[:breakPoint]

    # newLargeValTmax = largeInputsTmax[breakPoint:]
    # newLargeValTTP = largeInputsTTP[breakPoint:]
    # newLargeValADC = largeInputsADC[breakPoint:]
    # # newLargeValADCroi = largeInputsADCroi[breakPoint:]
    # newValFlair = outputsFlair[breakPoint:]


    return (largeInputsTmax,largeInputsTTP,largeInputsADC,smallInputsTmax,smallInputsTTP,smallInputsADC,outputsFlair)


def rot_Data(TMAX, TTP,ADC,tmax,ttp,adc,flair ):
    totalPoints,b,patchSize1,c=tmax.shape
    a,b,patchSize2,c=TMAX.shape
    newSmallInputsTmax = np.zeros((totalPoints * 4, 1, patchSize1, patchSize1))
    newLargeTrainTmax = np.zeros((totalPoints * 4, 1, patchSize2, patchSize2))
    newSmallInputsTTP = np.zeros((totalPoints * 4, 1, patchSize1, patchSize1))
    newLargeTrainTTP = np.zeros((totalPoints * 4, 1, patchSize2, patchSize2))
    newSmallInputsADC = np.zeros((totalPoints * 4, 1, patchSize1, patchSize1))
    newLargeTrainADC = np.zeros((totalPoints * 4, 1, patchSize2, patchSize2))
    newOutputsFlair = np.zeros((totalPoints * 4))
    for i in range(0, totalPoints):
        large1 = TMAX[i, 0, :, :]
        large2 = TTP[i, 0, :, :]
        large3 = ADC[i, 0, :, :]

        small1 = tmax[i, 0, :, :]
        small2 = ttp[i, 0, :, :]
        small3 = adc[i, 0, :, :]

        newOutputsFlair[4 * i] = flair[i]
        newOutputsFlair[4 * i + 1] = flair[i]
        newOutputsFlair[4 * i + 2] = flair[i]
        newOutputsFlair[4 * i + 3] = flair[i]

        newLargeTrainTmax[4 * i, 0, :, :] = large1
        newLargeTrainTmax[4 * i + 1, 0, :, :] = np.rot90(large1)
        newLargeTrainTmax[4 * i + 2, 0, :, :] = np.rot90(large1, 2)
        newLargeTrainTmax[4 * i + 3, 0, :, :] = np.rot90(large1, 3)

        newLargeTrainTTP[4 * i, 0, :, :] = large2
        newLargeTrainTTP[4 * i + 1, 0, :, :] = np.rot90(large2)
        newLargeTrainTTP[4 * i + 2, 0, :, :] = np.rot90(large2, 2)
        newLargeTrainTTP[4 * i + 3, 0, :, :] = np.rot90(large2, 3)

        newLargeTrainADC[4 * i, 0, :, :] = large3
        newLargeTrainADC[4 * i + 1, 0, :, :] = np.rot90(large3)
        newLargeTrainADC[4 * i + 2, 0, :, :] = np.rot90(large3, 2)
        newLargeTrainADC[4 * i + 3, 0, :, :] = np.rot90(large3, 3)

        newSmallInputsTmax[4 * i, 0, :, :] = small1
        newSmallInputsTmax[4 * i + 1, 0, :, :] = np.rot90(small1)
        newSmallInputsTmax[4 * i + 2, 0, :, :] = np.rot90(small1, 2)
        newSmallInputsTmax[4 * i + 3, 0, :, :] = np.rot90(small1, 3)

        newSmallInputsTTP[4 * i, 0, :, :] = small2
        newSmallInputsTTP[4 * i + 1, 0, :, :] = np.rot90(small2)
        newSmallInputsTTP[4 * i + 2, 0, :, :] = np.rot90(small2, 2)
        newSmallInputsTTP[4 * i + 3, 0, :, :] = np.rot90(small2, 3)

        newSmallInputsADC[4 * i, 0, :, :] = small3
        newSmallInputsADC[4 * i + 1, 0, :, :] = np.rot90(small3)
        newSmallInputsADC[4 * i + 2, 0, :, :] = np.rot90(small3, 2)
        newSmallInputsADC[4 * i + 3, 0, :, :] = np.rot90(small3, 3)

    return(newLargeTrainTmax,newLargeTrainTTP,newLargeTrainADC,newSmallInputsTmax,newSmallInputsTTP,newSmallInputsADC,newOutputsFlair)

##############################################################################################################
##############################################################################################################

def extract_patches_single(allPoints, totalPoints, fileIncluded, patchSize1, patchSize2):
    
    smallInputsTmax=np.zeros((totalPoints,1,patchSize2,patchSize2))
    largeInputsTmax=np.zeros((totalPoints,1,patchSize1,patchSize1))
    outputsFlair=np.zeros((totalPoints))

    for i in range(0,totalPoints):
        e,r,c,l = allPoints[i]
        smallInputsTmax[i,0] = fileIncluded[e, :, r-(patchSize2-1)/2 : r+(patchSize2+1)/2, c-(patchSize2-1)/2 : c+(patchSize2+1)/2]
        outputsFlair[i]=l
        largeInputsTmax[i,0] = fileIncluded[e, :, r-(patchSize1-1)/2 : r+(patchSize1+1)/2, c-(patchSize1-1)/2 : c+(patchSize1+1)/2]
       
    breakPoint = int(totalPoints)

    largeInputsTmax = largeInputsTmax[:breakPoint]
    outputsFlair = outputsFlair[:breakPoint]

    smallInputsTmax = smallInputsTmax[:breakPoint]
    return (largeInputsTmax,smallInputsTmax,outputsFlair)


def rot_Data_single(TMAX,tmax,flair):
    
    totalPoints,b,patchSize1,c=TMAX.shape
    a,b,patchSize2,c=tmax.shape
    
    newSmallInputsTmax = np.zeros((totalPoints * 4, 1, patchSize2, patchSize2))
    newLargeTrainTmax = np.zeros((totalPoints * 4, 1, patchSize1, patchSize1))
    newOutputsFlair = np.zeros((totalPoints * 4))
    
    for i in range(0, totalPoints):
        large1 = TMAX[i, 0, :, :]

        small1 = tmax[i, 0, :, :]

        newOutputsFlair[4 * i] = flair[i]
        newOutputsFlair[4 * i + 1] = flair[i]
        newOutputsFlair[4 * i + 2] = flair[i]
        newOutputsFlair[4 * i + 3] = flair[i]

        newLargeTrainTmax[4 * i, 0, :, :] = large1
        newLargeTrainTmax[4 * i + 1, 0, :, :] = np.rot90(large1)
        newLargeTrainTmax[4 * i + 2, 0, :, :] = np.rot90(large1, 2)
        newLargeTrainTmax[4 * i + 3, 0, :, :] = np.rot90(large1, 3)

        newSmallInputsTmax[4 * i, 0, :, :] = small1
        newSmallInputsTmax[4 * i + 1, 0, :, :] = np.rot90(small1)
        newSmallInputsTmax[4 * i + 2, 0, :, :] = np.rot90(small1, 2)
        newSmallInputsTmax[4 * i + 3, 0, :, :] = np.rot90(small1, 3)

    return(newLargeTrainTmax,newSmallInputsTmax,newOutputsFlair)

##############################################################################################################
##############################################################################################################

def iterate_minibatches(inputsBatch11, inputsBatch12, inputsBatch13, inputsBatch21, inputsBatch22, inputsBatch23, targets, batchsize, shuffle=False):
    assert len(inputsBatch11) == len(targets)
    if shuffle:
        indices = np.arange(len(inputsBatch11))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsBatch11) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputsBatch11[excerpt], inputsBatch12[excerpt], inputsBatch13[excerpt], inputsBatch21[excerpt], inputsBatch22[excerpt], inputsBatch23[excerpt], targets[excerpt]
