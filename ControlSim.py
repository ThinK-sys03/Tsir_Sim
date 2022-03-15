# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:26:34 2022

@author: Hanchu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from matplotlib.pyplot import MultipleLocator, FormatStrFormatter

def sis_model(y: tuple,
               N: int,
               beta: float,
               L: int,
               D: int,
               birth: float,
               changebeta: float):
    """
    

    Parameters
    ----------
    y : tuple
        Current states tSIR.
       y = [S, I]
       S: # susceptible individuals
       I: # infected individuals
    t : int
        Timestep.
    beta : float
        Transmission rate.
    L: float
        Infectious period (days).
    D: float
        Immune period (days).
    N : int
        Population size.
        
    Returns
    -------
    dydt: tuple, next state.

    """
    S, I = y
    
    dSdt = (N - S - I) / L - (beta * I * S) / N * changebeta + S  + birth
    
    dIdt = (beta * I * S) / N * changebeta - I / D + I

    dydt = (dSdt[0], dIdt[0])
    
    return dydt

if __name__ == '__main__':
    b = 0
    path = 'E:/NID_betas'
    files= os.listdir(path)
    Data = pd.read_csv('E:/NID_betas/RSV_parms.csv')
    # RealData = pd.read_excel('C:/Users/Hanchu/Desktop/Controlling.xlsx')
    ChangedBeta= np.loadtxt('D:/DesktopCollections/Desktopsave211222/FiguresForrebound/ChangedBeta.csv')
    NID = ['Influenza', 'RSV', 'Rhinovirus_Enterovirus', 'Adenovirus', 'Parainfluenza', 'Mycoplasma_Pneumoniae']
    Bed_usage = (310/10000, 1617/10000, 887/10000, 1072/10000, 734/10000, 2532/10000)
    # Data for influx
    RH = pd.read_excel('E:\HumidityHK.xlsx', sheet_name = 'Sheet2').values
    temperature = pd.read_excel('E:\HumidityHK.xlsx', sheet_name = 'Sheet3').values
    # RH = humidity.values
    es = 611.2 * np.exp(17.67*temperature/(temperature+243.5))
    e = RH/100*es
    SH = (1-0.378) * e/(101325-(0.378 * e))
    Dw = 18.016*es*(RH/100) / ((temperature+273.1)*8.314472)
    R0 = np.exp(-180 * SH + np.log(3 - 1.2)) + 1.2
    
    simulationyear = 58
    times = np.arange(0, simulationyear, 1/52)
    controlWeekStart = 8
    controlWeekLength = 130
    controlStart = 50*52 + controlWeekStart
    controlEnd = 50*52 + controlWeekStart + controlWeekLength
    pop = 7200000
    births = 1000
    nround = 1
    stochastic = 0
    
    I0 = 0.3*pop
    S0 = 0.3*pop
    alpha = 0.97
    
    Istates = []
    MaxINumber = 10000
    betaData = []
    betaData.append(R0[:,0])
    for i in NID:
        Data = pd.read_csv('E:/NID_betas/'+i+'_parms.csv')
        betaData.append(Data['beta'].values)
    del betaData[1]
    # plt.figure(figsize=(18, 16))
    BedUseState = []
    ThresholdSet = []
    for m in tqdm(np.arange(0.1, 1 + 0.01, 0.01)): # pick threshold
        for n in np.arange(0.3, 1, 0.01): # pick control beta
            CumlativeIData = []
            CumlativeSData = []
            birthsdaily = births/7
            D = 4
            L = 40 * 7
            State_influx = []
            dydt = [S0, I0]
            State_influx.append(I0)
            I_State_other = np.zeros([simulationyear*52+1, 6])
            S_State_other = np.zeros([simulationyear*52+1, 6])
            I_State_other[0,:] = I0
            S_State_other[0,:] = S0
            for t in range(simulationyear*52): # simulate according to time
                if t < controlStart:
                    changebeta = 1
                    for i in np.arange(6):
                        if i == 0:
                            beta = R0[t%52]/D
                            for k in range(7):
                                dydt = sis_model(dydt, pop, beta, L, D, birthsdaily, changebeta) 
                            State_influx.append(dydt[1])
                        else:
                            beta = betaData[i][t%52]
                            lambd = min(S_State_other[t,i], pow(beta * S_State_other[t,i] * I_State_other[t, i], alpha))
                            I_State_other[t+1, i] = lambd
                            S_State_other[t+1, i] = max(S_State_other[t,i] + births - I_State_other[t+1, i], 0)
                elif (t >= (controlStart)) & (t <= controlEnd):
                    for i in np.arange(6):
                        if i == 0:
                            beta = R0[t%52]/D
                            changebeta = pow(0.1, 1/7)
                            for k in range(7):
                                dydt = sis_model(dydt, pop, beta, L, D, birthsdaily, changebeta)
                            State_influx.append(dydt[1])
                        else:
                            beta = betaData[i][t%52]
                            changebeta = 1 - ChangedBeta[i]
                            lambd = min(S_State_other[t,i], pow(beta * S_State_other[t,i] * (I_State_other[t, i] * changebeta), alpha))
                            I_State_other[t+1, i] = lambd
                            S_State_other[t+1, i] = max(S_State_other[t,i] * (1-(89200 + 50400)/7200000) + births * 0.6 - I_State_other[t+1, i], 0)
                elif t > controlEnd:
                    for i in np.arange(6):
                        if i == 0:
                            beta = R0[t%52]/D
                            if np.sum(I_State_other[t, :]*Bed_usage) + (State_influx[-1]*Bed_usage[0]) > m*MaxINumber:
                                changebeta = pow(n, 1/7)
                            else:
                                changebeta = 1
                            for k in range(7):
                                dydt = sis_model(dydt, pop, beta, L, D, birthsdaily, changebeta)
                            State_influx.append(dydt[1])
                        else:
                            beta = betaData[i][t%52]
                            if np.sum(I_State_other[t, :]*Bed_usage) + (State_influx[-1]*Bed_usage[0]) > m*MaxINumber:
                                changebeta = n
                                lambd = min(S_State_other[t,i], pow(beta * S_State_other[t,i] * (I_State_other[t, i] * changebeta), alpha))
                                I_State_other[t+1, i] = lambd
                                S_State_other[t+1, i] = max(S_State_other[t,i] * (1-(89200 + 50400)/7200000) + births * 0.6 - I_State_other[t+1, i], 0)
                            else:
                                lambd = min(S_State_other[t,i], pow(beta * S_State_other[t,i] * (I_State_other[t, i]), alpha))
                                I_State_other[t+1, i] = lambd
                                S_State_other[t+1, i] = max(S_State_other[t,i] + births - I_State_other[t+1, i], 0)
                else:
                    print('more time:', t)
            influx = np.array(State_influx)
            I_State = np.c_[influx, I_State_other]
            I_State = np.delete(I_State, 1, axis=1)
            #I_State_test = I_State*Bed_usage
            I_aggregated = np.sum(I_State*Bed_usage, axis=1)
            BedUseState.append(I_aggregated[controlEnd:])
            ThresholdSet.append((np.round(m, decimals = 2), np.round(n, decimals = 2)))
            sumI = np.sum(I_aggregated[controlEnd:])
            MaxI = max(I_aggregated[controlEnd:])
            meanI = np.average(I_aggregated[controlEnd:])
            StdI = np.std(I_aggregated[controlEnd:])
            exceeddays = np.sum(I_aggregated[controlEnd:]>(14976*0.5))
            exceednumber = np.zeros(len(I_aggregated[controlEnd:]))
            exceedloc = I_aggregated[controlEnd:]
            exceednumber[exceedloc-(14976*0.5)>0] = exceedloc[exceedloc-(14976*0.5)>0] - (14976*0.5)
            exceednumber = np.sum(exceednumber)
            Istates.append((np.round(m, decimals = 2), np.round(n, decimals = 2), sumI, MaxI, meanI, StdI, exceeddays, exceednumber))
    
    OutputBed = np.array(BedUseState)   
    plotsensitive = [0.1, 0.3, 0.5, 0.7, 0.9]
    count = 1
    plt.figure(figsize=(36, 48))
    for i in plotsensitive:
        exceedweeks = np.sum(OutputBed>(14976*i), axis = 1)
        updateexceednum = np.zeros((len(OutputBed), np.size(OutputBed, axis=1)))
        updateexceednum[OutputBed-(14976*i)>0] = OutputBed[OutputBed-(14976*i)>0] - (14976*i)
        updateexceednum = np.sum(updateexceednum, axis=1)
        forplot = np.c_[np.array(ThresholdSet), exceedweeks, updateexceednum]
        Istate = pd.DataFrame(forplot, columns = ['Threshold', 'Transmission rate', 'OverloadWeek', 'OverloadNum'])
        plt.subplot(len(plotsensitive), 2, count)
        plotI = Istate.pivot('Threshold', 'Transmission rate', 'OverloadWeek')
        sns.heatmap(plotI, cmap='Spectral_r', fmt='d', xticklabels =5, yticklabels =5)
        plt.title('Overload Infections for ' + str(i) + ' of total beds have been occupied', fontsize=24)
        count += 1
        plt.subplot(len(plotsensitive), 2, count)
        ax = plt.gca()
        plotI = Istate.pivot('Threshold', 'Transmission rate', 'OverloadNum') 
        sns.heatmap(plotI, cmap='Greens', fmt='d', xticklabels =5, yticklabels =5)
        plt.title('Overload Weeks for ' + str(i) + ' of total beds have been occupied', fontsize=24)
        count += 1
        # plt.xticks([30])
    plt.tight_layout()
    plt.savefig('sensitive_overload_details_Overload_Update.jpg', dpi = 600)
    
    # AvailableCount = []
    # for i in np.arange(0.01, 1.01, 0.01):
    #     exceedweeks = np.sum(OutputBed > (14976*i), axis = 1)
    #     AvailPlan = np.sum(exceedweeks == 0)
    #     AvailableCount.append((np.around(i, decimals = 2), AvailPlan))
    # AvailableCount = np.array(AvailableCount)
    # plt.axvspan(0.01, 0.35, color='red', alpha=0.3, lw=0)
    # plt.axvspan(0.95, 1.00, color='green', alpha=0.3, lw=0)
    # # plt.figure(figsize=(36, 48))
    # plt.plot(AvailableCount[:,0], AvailableCount[:,1]/len(OutputBed), color='blue')
    # # plt.legend()    
    # plt.text(0.36, 0.6, 'All plans are unavailable', fontdict={'size':'8','color':'r'})
    # plt.text(0.70, 0.2, 'All plans are available', fontdict={'size':'8','color':'g'})
    # plt.xlabel('Percentage of unoccupied hospital beds')
    # plt.ylabel('Percentage of available countermeasures with fixed threshold', fontsize = 8)
    # plt.tight_layout()
    # plt.savefig('Count_of_available_plans_Overload_Update.jpg', dpi = 600)
    # # Istates = np.array(Istates)
    # Istates = pd.DataFrame(Istates, columns = ['Threshold', 'Transmission rate', 'Sum', 'Max', 'Mean', 'Std', 'Overload', 'OverloadNum'])
    # plt.figure(figsize=(18, 8))
    # plt.subplot(1, 2, 1)
    # plotI = Istates.pivot('Threshold', 'Transmission rate', 'OverloadNum')
    # sns.heatmap(plotI, cmap='Spectral_r', fmt='d')
    # plt.title('Overload Infections')
    # plt.subplot(1, 2, 2)
    # plotI = Istates.pivot('Threshold', 'Transmission rate', 'Overload') 
    # sns.heatmap(plotI, cmap='Greens', fmt='d')
    # plt.title('Overload Weeks')
    # plt.tight_layout()
    # # fig = plt.get_figure()
    # plt.savefig('sensitive_overload_details_OverloadNum_Update_50.jpg', dpi = 600)
    
    # # OutputBed = np.array(BedUseState)
    # OutputThreshold = np.array(ThresholdSet)
    # OutputDetails = np.c_[OutputThreshold, OutputBed]
    # # np.savetxt('OutputDetails.csv', OutputDetails,delimiter=',')