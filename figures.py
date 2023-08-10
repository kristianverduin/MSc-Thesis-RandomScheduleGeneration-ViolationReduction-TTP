import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn import preprocessing
from scipy import interpolate

def createAverages():
    for i in range(4, 51, 2):
        file = open("D:\MasterThesis\Experiment2\ViolationExperiment\Violations" + str(i) + ".txt", "r")
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[1]
        repeat = text[2]
        robin1 = text[3]
        robin2 = text[4]
        robin3 = text[5]

        avgHA = np.average(homeAway)
        maxHA = np.max(homeAway)
        minHA = np.min(homeAway)
        stdHA = np.std(homeAway)
        medianHA = np.median(homeAway)

        homeAway = [avgHA, maxHA, minHA, stdHA, medianHA]

        avgRepeat = np.average(repeat)
        maxRepeat = np.max(repeat)
        minRepeat = np.min(repeat)
        stdRepeat = np.std(repeat)
        medianRepeat = np.median(repeat)

        repeat = [avgRepeat, maxRepeat, minRepeat, stdRepeat, medianRepeat]

        avgRobin1 = np.average(robin1)
        maxRobin1 = np.max(robin1)
        minRobin1 = np.min(robin1)
        stdRobin1 = np.std(robin1)
        medianRobin1 = np.median(robin1)

        robin1 = [avgRobin1, maxRobin1, minRobin1, stdRobin1, medianRobin1]

        avgRobin2 = np.average(robin2)
        maxRobin2 = np.max(robin2)
        minRobin2 = np.min(robin2)
        stdRobin2 = np.std(robin2)
        medianRobin2 = np.median(robin2)

        robin2 = [avgRobin2, maxRobin2, minRobin2, stdRobin2, medianRobin2]

        avgRobin3 = np.average(robin3)
        maxRobin3 = np.max(robin3)
        minRobin3 = np.min(robin3)
        stdRobin3 = np.std(robin3)
        medianRobin3 = np.median(robin3)

        robin3 = [avgRobin3, maxRobin3, minRobin3, stdRobin3, medianRobin3]

        file = open("D:\MasterThesis\Experiment2\ViolationExperiment\Changed\ViolationsAvg" + str(i) + ".txt", "w")
        np.savetxt(file, [homeAway, repeat, robin1, robin2, robin3], delimiter=',')
        file.close()

def makeGraphs():
    avgHA = []
    stdHA = []
    maxHA = []
    minHA = []

    avgRepeat = []
    stdRepeat = []
    maxRepeat = []
    minRepeat = []

    avgRobin = []
    stdRobin = []
    maxRobin = []
    minRobin = []

    avgTotal = []
    stdTotal = []
    maxTotal = []
    minTotal = []

    labels = []
    
    for i in range(4, 51, 2):
        labels.append(i)
        homeAway = []
        repeat = []
        robin = []
        total = []

        for j in range(5):
            file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/' + str(j) + '/Violations' + str(i) + '.txt', "r")
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway.extend(text[0])
            repeat.extend(text[1])
            robin.extend(text[2])
            for z in range(len(text[0])):
                total.append(text[0][z] + text[1][z] + text[2][z])

        avgHA.append(np.mean(homeAway))
        stdHA.append(np.std(homeAway))
        maxHA.append(max(homeAway))
        minHA.append(min(homeAway))

        avgRepeat.append(np.mean(repeat))
        stdRepeat.append(np.std(repeat))
        maxRepeat.append(max(repeat))
        minRepeat.append(min(repeat))

        avgRobin.append(np.mean(robin))
        stdRobin.append(np.std(robin))
        maxRobin.append(max(robin))
        minRobin.append(min(robin))

        avgTotal.append(np.mean(total))
        stdTotal.append(np.std(total))
        maxTotal.append(max(total))
        minTotal.append(min(total))

        print(i)
        print(min(total))
        print()

    print(minTotal)
    exit()

    plt.rcParams.update({'font.size': 25})

    fig, (ax1, ax2) = plt.subplots(1, 2)

    stdRobin1 = []
    stdRobin2 = []
    stdRepeat1 = []
    stdRepeat2 = []
    stdHomeAway1 = []
    stdHomeAway2 = []
    stdTotal1 = []
    stdTotal2 = []

    for j in range(len(avgRobin)):
        stdRobin1.append(avgRobin[j] + stdRobin[j])
        stdRobin2.append(avgRobin[j] - stdRobin[j])

        stdRepeat1.append(avgRepeat[j] + stdRepeat[j])
        stdRepeat2.append(avgRepeat[j] - stdRepeat[j])

        stdHomeAway1.append(avgHA[j] + stdHA[j])
        stdHomeAway2.append(avgHA[j] - stdHA[j])

        stdTotal1.append(avgTotal[j] + stdTotal[j])
        stdTotal2.append(avgTotal[j] - stdTotal[j])

    four = ax1.fill_between(labels, maxTotal, minTotal, alpha=0.5, color='Green')
    three = ax1.fill_between(labels, maxRobin, minRobin, alpha=0.5, color='Red')
    one = ax1.fill_between(labels, maxHA, minHA, alpha=0.5, color='Blue')
    two = ax1.fill_between(labels, maxRepeat, minRepeat, alpha=0.5, color='Orange')

    eight = ax1.plot(labels, avgTotal, c='Black')
    seven, = ax1.plot(labels, avgRobin, c='Black')
    five, = ax1.plot(labels, avgHA, c='Black')
    six, = ax1.plot(labels, avgRepeat, c='Black')

    ax2.plot(labels, avgTotal, c='Green')
    ax2.plot(labels, avgRobin, c='Red')
    ax2.plot(labels, avgHA, c='Blue')
    ax2.plot(labels, avgRepeat, c='Orange')

    fitHA = np.poly1d(np.polyfit(labels, avgHA, 2))
    print(fitHA)
    MSEHA = np.square(np.subtract(avgHA, fitHA(labels))).mean()
    rmseHA = math.sqrt(MSEHA)
    print(rmseHA)

    fitRepeat = np.poly1d(np.polyfit(labels, avgRepeat, 2))
    print(fitRepeat)
    MSERepeat = np.square(np.subtract(avgRepeat, fitRepeat(labels))).mean()
    rmseRepeat = math.sqrt(MSERepeat)
    print(rmseRepeat)

    fitRobin = np.poly1d(np.polyfit(labels, avgRobin, 2))
    print(fitRobin)
    MSERobin = np.square(np.subtract(avgRobin, fitRobin(labels))).mean()
    rmseRobin = math.sqrt(MSERobin)
    print(rmseRobin)

    fitTotal = np.poly1d(np.polyfit(labels, avgTotal, 2))
    print(fitTotal)
    MSETotal = np.square(np.subtract(avgTotal, fitTotal(labels))).mean()
    rmseTotal = math.sqrt(MSETotal)
    print(rmseTotal)

    coefHA = []
    for i in fitHA.coefficients:
        coefHA.append(round(i, 2))

    coefRepeat = []
    for i in fitRepeat.coefficients:
        coefRepeat.append(round(i, 2))

    coefRobin = []
    for i in fitRobin.coefficients:
        coefRobin.append(round(i, 2))

    coefTotal = []
    for i in fitTotal.coefficients:
        coefTotal.append(round(i, 2))

    totalString = str(coefTotal[0]) + 'x$^2$ - ' + str(coefTotal[1])[1:] + 'x - ' + str(coefTotal[2])[1:]

    robinString = str(coefRobin[0]) + 'x$^2$ - ' + str(coefRobin[1])[1:] + 'x - ' + str(coefRobin[2])[1:]

    HAString = str(coefHA[0]) + 'x$^2$ - ' + str(coefHA[1])[1:] + 'x'
    
    repeatString = str(coefRepeat[1]) + 'x - ' + str(coefRepeat[2])[1:]

    labels2 = np.linspace(4, 100, 49)

    twelve, = ax2.plot(labels2, fitTotal(labels2), color='Green', ls='--', dashes=(5, 5))
    eleven, = ax2.plot(labels2, fitRobin(labels2), color='Red', ls='--', dashes=(5, 5))
    nine, = ax2.plot(labels2, fitHA(labels2), color='Blue',ls='--', dashes=(5, 5))
    ten, = ax2.plot(labels2, fitRepeat(labels2), color='Orange',ls='--', dashes=(5, 5))

    test = ax1.legend([four, three, one, two, six, four, five, eight], ['Total', 'Double Round-Robin', 'maxStreak', 'noRepeat', 'Average'], edgecolor='black')
    test2 = ax2.legend([twelve, eleven, nine, ten], [totalString, robinString, HAString, repeatString], edgecolor='black')

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')
    test2.get_frame().set_alpha(0.5)
    test2.get_frame().set_facecolor('wheat')

    ax2.yaxis.tick_right()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.xticks([0, 10, 20, 30, 40, 50])

    plt.xlabel('Number of Teams')
    plt.ylabel('Number of Violations')

    plt.plot()
    plt.show()

def checkValid():
    for i in range(4, 51, 2):
        file = open("D:\MasterThesis\Experiment2\ViolationExperiment\Violations\Violations" + str(i) + ".txt", "r")
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[1]
        repeat = text[2]
        robin = text[3]
        robin2 = text[4]
        robin3 = text[5]

        nrCorrect = 0

        total = []

        for j in range(len(homeAway)):
            if homeAway[j] == 0 and repeat[j] == 0 and robin[j] == 0 and robin2[j] == 0 and robin3[j] == 0:
                nrCorrect += 1
            total.append(homeAway[j]+repeat[j]+robin[j]+robin2[j]+robin3[j])
        
        print(min(total))
    
        file = open(r"D:\MasterThesis\Experiment2\ViolationExperiment\Violations\correct\testing" + str(i) + ".txt", "w")
        np.savetxt(file, [nrCorrect], fmt='%i',delimiter=',')
        file.close()

def testing():
        for i in range(4, 51, 2):
            file = open(r"D:\MasterThesis\Experiment2\ViolationExperiment\Violations\Changed\ViolationsAvg" + str(i) + ".txt", "r")
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway = text[0]
            repeat = text[1]
            robin = text[2]

            with open(r"D:\MasterThesis\Experiment2\ViolationExperiment\Violations\Changed\ViolationsAvg" + str(i) + ".csv", "w") as file:
                writer = csv.writer(file)

                writer.writerow(['Constraint', 'Average', 'Maximum', 'Minimum', 'Standard deviation', 'Median'])
                writer.writerow(['maxStreak', homeAway[0], homeAway[1], homeAway[2], homeAway[3], homeAway[4]])
                writer.writerow(['noRepeat', repeat[0], repeat[1], repeat[2], repeat[3], repeat[4]])
                writer.writerow(['Double round-robin', robin[0], robin[1], robin[2], robin[3], robin[4]])

                file.close()

def makeMinViolationGraphs():
    for j in range(4, 51, 2):
        totals = []
        HA = []
        repeat = []
        robin = []
        for i in range(5):
            folder = 'E:/NewExperiment/NewResults/MinViolationExperiment2/' + str(i) + '/'
            file = folder + 'ViolationsMin' + str(j) + '.txt'
            file = open(file, 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            HA.append(text[0])
            repeat.append(text[1])
            robin.append(text[2])
            totals.append(text[3])

        medianTotal = []
        medianHA = []
        medianRepeat = []
        medianRobin = []
        for i in range(len(totals[0])):
            medianTotal.append(np.mean([totals[0][i], totals[1][i], totals[2][i], totals[3][i], totals[4][i]]))
            medianHA.append(np.mean([HA[0][i], HA[1][i], HA[2][i], HA[3][i], HA[4][i]]))
            medianRepeat.append(np.mean([repeat[0][i], repeat[1][i], repeat[2][i], repeat[3][i], repeat[4][i]]))
            medianRobin.append(np.mean([robin[0][i], robin[1][i], robin[2][i], robin[3][i], robin[4][i]]))

        file = open("E:/NewExperiment/NewResults/MinViolationExperiment2/Mean" + str(j) + ".txt", "w")
        np.savetxt(file, [medianHA, medianRepeat, medianRobin, medianTotal], delimiter=',')
        file.close()

def medianGraphs():
    x = []
    for i in range(1, 1000001):
        x.append(i)
    x = np.array(x)
    for i in range(10, 51, 2):
        file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Mean/Mean' + str(i) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        hA = text[0]
        repeat = text[1]
        robin = text[2]
        total = text[3]

        file2 = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Mean/Mean' + str(50) + '.txt', 'r')
        text2 = np.loadtxt(file2, delimiter=',')
        file2.close()

        hA2 = text2[0]
        repeat2 = text2[1]
        robin2 = text2[2]
        total2 = text2[3]

        def func(x, a, b):
            return a+b*np.log(x)
                
        poptHA, pcov = curve_fit(func, x, hA, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRepeat, pcov = curve_fit(func, x, repeat, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRobin, pcov = curve_fit(func, x, robin, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptTotal, pcov = curve_fit(func, x, total, bounds=((-np.inf, -np.inf), (np.inf, 0)))

        poptHA2, pcov2 = curve_fit(func, x, hA2, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRepeat2, pcov2 = curve_fit(func, x, repeat2, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRobin2, pcov2 = curve_fit(func, x, robin2, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptTotal2, pcov2 = curve_fit(func, x, total2, bounds=((-np.inf, -np.inf), (np.inf, 0)))

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(robin, func(x, *poptRobin))).mean()
        rmseRobin  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(hA, func(x, *poptHA))).mean()
        rmseHA  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(repeat, func(x, *poptRepeat))).mean()
        rmseRepeat  = math.sqrt(MSETotal)
        #print(rmseTotal)

        print(rmseHA)
        print(rmseRepeat)
        print(rmseRobin)
        print(rmseTotal)
        
        MSETotal  = np.square(np.subtract(total2, func(x, *poptTotal2))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(robin2, func(x, *poptRobin2))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(hA2, func(x, *poptHA2))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        MSETotal  = np.square(np.subtract(repeat2, func(x, *poptRepeat2))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        plt.rcParams.update({'font.size': 25})

        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        one = ax1.plot(x, total, label='Total', color='Green')
        two = ax1.plot(x, robin, label='Double Round-Robin', color='Red')
        three = ax1.plot(x, hA, label='maxStreak', color='Blue')
        four = ax1.plot(x, repeat, label='noRepeat', color='Orange')

        five = ax1.plot(x, func(x, *poptTotal), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))
        six = ax1.plot(x, func(x, *poptRobin), label='Double Round-Robin Fitted', color='Red', ls='--', dashes=(5, 5))
        seven = ax1.plot(x, func(x, *poptHA), label='maxStreak Fitted', color='Blue', ls='--', dashes=(5, 5))
        eight = ax1.plot(x, func(x, *poptRepeat), label='noRepeat Fitted', color='Orange', ls='--', dashes=(5, 5))

        ax2.plot(x, total2, label='Total', color='Green')
        ax2.plot(x, robin2, label='Double Round-Robin', color='Red')
        ax2.plot(x, hA2, label='maxStreak', color='Blue')
        ax2.plot(x, repeat2, label='noRepeat', color='Orange')

        ax2.plot(x, func(x, *poptTotal2), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))
        ax2.plot(x, func(x, *poptRobin2), label='Double Round-Robin Fitted', color='Red', ls='--', dashes=(5, 5))
        ax2.plot(x, func(x, *poptHA2), label='maxStreak Fitted', color='Blue', ls='--', dashes=(5, 5))
        ax2.plot(x, func(x, *poptRepeat2), label='noRepeat Fitted', color='Orange', ls='--', dashes=(5, 5))

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax1.xaxis.offsetText.set_visible(False)
        ax2.xaxis.offsetText.set_visible(False)

        ax2.yaxis.tick_right()

        a = str(poptTotal[1])[:4] + ' ln(x) + ' + str(poptTotal[0])[:5]
        b = str(poptRobin[1])[:4] + ' ln(x) + ' + str(poptRobin[0])[:2]
        c = str(poptHA[1])[:4] + ' ln(x) + ' + str(poptHA[0])[:2]
        d = str(poptRepeat[1])[:4] + ' ln(x) + ' + str(poptRepeat[0])[:3]

        e = str(poptTotal2[1])[:5] + 'ln(x) + ' + str(poptTotal2[0])[:6]
        f = str(poptRobin2[1])[:5] + 'ln(x) + ' + str(poptRobin2[0])[:6]
        g = str(poptHA2[1])[:4] + 'ln(x) + ' + str(poptHA2[0])[:5]
        h = str(poptRepeat2[1])[:4] + 'ln(x) + ' + str(poptRepeat2[0])[:4]



        props = dict(boxstyle='round', facecolor='Green', alpha=0.2)
        plt.text(0.85, 0.62, a, horizontalalignment='center', verticalalignment='top', transform = ax1.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Red', alpha=0.2)
        plt.text(0.875, 0.44, b, horizontalalignment='center', verticalalignment='top', transform = ax1.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Blue', alpha=0.2)
        plt.text(0.875, 0.12, c, horizontalalignment='center', verticalalignment='top', transform = ax1.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Orange', alpha=0.2)
        plt.text(0.872, 0.04, d, horizontalalignment='center', verticalalignment='top', transform = ax1.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Green', alpha=0.2)
        ax2.text(0.83, 0.908, e, horizontalalignment='center', verticalalignment='top', transform = ax2.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Red', alpha=0.2)
        ax2.text(0.83, 0.7, f, horizontalalignment='center', verticalalignment='top', transform = ax2.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Blue', alpha=0.2)
        ax2.text(0.85, 0.2, g, horizontalalignment='center', verticalalignment='top', transform = ax2.transAxes, bbox=props, fontsize=14)

        props = dict(boxstyle='round', facecolor='Orange', alpha=0.2)
        ax2.text(0.86, 0.09, h, horizontalalignment='center', verticalalignment='top', transform = ax2.transAxes, bbox=props, fontsize=14)

        #ax1.legend(['_Total_', '_DRR_', '_MS_', '_NR_', a, b, c, d], loc='upper center')
        #ax2.legend(['_Total_', '_DRR_', '_MS_', '_NR_', e, f, g, h], loc='center')

        ax1.text(0.5, 1.01, str(i) + ' Teams', fontweight='bold', horizontalalignment='center', transform = ax1.transAxes,)
        ax2.text(0.5, 1.01, '50 Teams', fontweight='bold', horizontalalignment='center', transform=ax2.transAxes)

        test = fig.legend(['Total', 'Double Round-Robin', 'maxStreak', 'noRepeat'], loc='upper center', ncols=4, edgecolor='black') 

        test.get_frame().set_alpha(0.5)
        test.get_frame().set_facecolor('wheat')

        plt.xlabel('Schedules Generated (10$^6$)')
        plt.ylabel('Number of Violations')

        plt.subplots_adjust(wspace=0.02, hspace=0.1)

        plt.show()

        exit()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.plot(x, total, color='Green')
        ax1.plot(x, robin, color='Red')

        ax2.plot(x, total, color='Green')
        ax2.plot(x, hA, color='Blue')

        ax3.plot(x, total, color='Green')
        ax3.plot(x, repeat, color='Orange')

        plt.xlabel('Number of Teams')
        plt.ylabel('a')

        plt.show()

        exit()

def getParameters():
    x = []
    for i in range(1, 1000001):
        x.append(i)
    x = np.array(x)
    for i in range(4, 51, 2):
        file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Mean/Mean' + str(i) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        hA = text[0]
        repeat = text[1]
        robin = text[2]
        total = text[3]

        def func(x, a, b):
            return a+b*np.log(x)
                
        poptHA, pcov = curve_fit(func, x, hA, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRepeat, pcov = curve_fit(func, x, repeat, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptRobin, pcov = curve_fit(func, x, robin, bounds=((-np.inf, -np.inf), (np.inf, 0)))
        poptTotal, pcov = curve_fit(func, x, total, bounds=((-np.inf, -np.inf), (np.inf, 0)))

        aHA = poptHA[0]
        bHA = poptHA[1]
        interceptHA = np.exp(-aHA/bHA)

        aRepeat = poptRepeat[0]
        bRepeat = poptRepeat[1]
        interceptRepeat = np.exp((0.5-aRepeat)/bRepeat)
        print(i)

        aRobin = poptRobin[0]
        bRobin = poptRobin[1]
        interceptRobin = np.exp((0.5-aRobin)/bRobin)

        aTotal = poptTotal[0]
        bTotal = poptTotal[1]
        interceptTotal = np.exp((0.5-aTotal)/bTotal)

        print(interceptTotal)

        file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Parameters3/Parameters' + str(i) + '.txt', 'w')
        np.savetxt(file, [aHA, bHA, interceptHA, aRepeat, bRepeat, interceptRepeat, aRobin, bRobin, interceptRobin, aTotal, bTotal, interceptTotal], delimiter=',')
        file.close()

def gaus(X,C,X_mean,sigma):
    return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

def makeBellCurves(z):
    x = []
    for i in range(1, 1000001):
        x.append(i)
    x = np.array(x)

    homeAway = []
    repeat = []
    robin1 = []

    for i in range(5):
        file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/' + str(i) + '/Violations' + str(z) + '.txt', "r")
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway.extend(text[0])
        repeat.extend(text[1])
        robin1.extend(text[2])

    total = []
    for i in range(len(homeAway)):
        total.append(homeAway[i]+repeat[i]+robin1[i])
    total = np.array(total)

    plt.rcParams.update({'font.size': 15})

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

    weights = np.ones_like(homeAway)/float(len(homeAway))
    hist, bins, _ = ax2.hist(homeAway, bins=30, weights=weights, color='Blue', edgecolor='black', linewidth=1.2)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bins[ii+1]+bins[ii])/2
    
    y_hist=hist
    mean = sum(x_hist*y_hist)/sum(y_hist)
    sigma = np.std(homeAway)

    popt, pcov = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist),mean,sigma], maxfev=5000, bounds=((0, -np.inf, -np.inf), (1, np.inf, np.inf)))
    x_hist_2=np.linspace(np.min(x_hist)-9,np.max(x_hist),100)
    ax2.plot(x_hist_2, gaus(x_hist_2,*popt),'k', linewidth=2)
    ax2.text(0.03, 0.9, 'maxStreak', fontweight='bold', horizontalalignment='left', transform=ax2.transAxes)
    ax2.set_xlim(np.min(x_hist)-10, np.max(x_hist)+10)
    ax2.set_ylim(0, 0.2)

    mse  = np.square(np.subtract(y_hist, gaus(x_hist, *popt))).mean()
    rmse  = math.sqrt(mse)
    #rmse= round(rmse)
    print(rmse)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.975, 0.95, u"\u03bc = " + str(round(mean, 3)) + '\n' + u"\u03c3 = " + str(round(sigma, 3)), horizontalalignment='right', verticalalignment='top', transform = ax2.transAxes, bbox=props)


    weights = np.ones_like(repeat)/float(len(repeat))
    hist, bins, _ = ax3.hist(repeat, bins=30, weights=weights, label='noRepeat', color='Orange', edgecolor='black', linewidth=1.2)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bins[ii+1]+bins[ii])/2

    y_hist=hist
    mean = sum(x_hist*y_hist)/sum(y_hist)                  
    sigma = np.std(repeat) 

    popt, pcov = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist),mean,sigma])
    x_hist_2=np.linspace(np.min(x_hist)-9,np.max(x_hist),100)
    ax3.plot(x_hist_2,gaus(x_hist_2,*popt),'k', linewidth=2)
    ax3.text(0.03, 0.9, 'noRepeat', fontweight='bold', horizontalalignment='left', transform=ax3.transAxes)
    ax3.set_xlim(np.min(x_hist)-10, np.max(x_hist)+10)
    ax3.set_ylim(0, 0.2)

    mse  = np.square(np.subtract(y_hist, gaus(x_hist, *popt))).mean()
    rmse  = math.sqrt(mse)
    #rmse= round(rmse)
    print(rmse)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.975, 0.95, u"\u03bc = " + str(round(mean, 3)) + '\n' + u"\u03c3 = " + str(round(sigma, 3)), horizontalalignment='right', verticalalignment='top', transform = ax3.transAxes, bbox=props)


    weights = np.ones_like(robin1)/float(len(robin1))
    hist, bins, _ = ax1.hist(robin1, bins=30, weights=weights, label='Double Round-Robin', color='Red', edgecolor='black', linewidth=1.2)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bins[ii+1]+bins[ii])/2

    y_hist=hist
    mean = sum(x_hist*y_hist)/sum(y_hist)                  
    sigma = np.std(robin1)

    popt, pcov = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist),mean,sigma])
    x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),100)
    ax1.plot(x_hist_2,gaus(x_hist_2,*popt),'k', linewidth=2)
    ax1.text(0.03, 0.9, 'Double Round-Robin', fontweight='bold', horizontalalignment='left', transform=ax1.transAxes)
    ax1.set_xlim(np.min(x_hist)-10, np.max(x_hist)+10)
    ax1.set_ylim(0, 0.2)

    mse  = np.square(np.subtract(y_hist, gaus(x_hist, *popt))).mean()
    rmse  = math.sqrt(mse)
    #rmse= round(rmse)
    print(rmse)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.975, 0.95, u"\u03bc = " + str(round(mean, 3)) + '\n' + u"\u03c3 = " + str(round(sigma, 3)), horizontalalignment='right', verticalalignment='top', transform = ax1.transAxes, bbox=props)

    weights = np.ones_like(total)/float(len(total))
    hist, bins, _ = ax0.hist(total, bins=30, weights=weights, label='Total', color='Green', edgecolor='black', linewidth=1.2)

    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bins[ii+1]+bins[ii])/2

    y_hist=hist
    mean = sum(x_hist*y_hist)/sum(y_hist)                  
    sigma = np.std(total)

    popt, pcov = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist),mean,sigma])
    x_hist_2=np.linspace(np.min(0), np.max(x_hist)+np.min(x_hist), int((np.max(x_hist)+np.min(x_hist))*3))
    ax0.plot(x_hist_2,gaus(x_hist_2,*popt),'k', linewidth=2)
    ax0.text(0.03, 0.9, 'Total', fontweight='bold', horizontalalignment='left', transform=ax0.transAxes)
    ax0.set_xlim(np.min(x_hist)-10, np.max(x_hist)+10)
    ax0.set_ylim(0, 0.2)

    mse  = np.square(np.subtract(y_hist, gaus(x_hist, *popt))).mean()
    rmse  = math.sqrt(mse)
    #rmse= round(rmse, 10)
    print(rmse)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax0.text(0.975, 0.95, u"\u03bc = " + str(round(mean, 3)) + '\n' + u"\u03c3 = " + str(round(sigma, 3)), horizontalalignment='right', verticalalignment='top', transform = ax0.transAxes, bbox=props)

    #cdf = np.cumsum(gaus(x_hist_2, *popt))
    #cdf = cdf/np.max(cdf)

    #ax3.plot(x_hist_2, cdf, 'r')

    ax1.yaxis.tick_right()
    ax3.yaxis.tick_right()

    whole = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.xlabel('Violations')
    fig.text(0.06, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=16)

    plt.subplots_adjust(wspace=0.02, hspace=0.1)

    fig.set_size_inches(19.2, 10.8)

    fig.savefig('C:/Users/megki_000/Desktop/Hist' + str(z) + '.png')
    plt.show()

def getChances():
    totalChance = []

    for j in range(4, 51, 2):
        print(j)
        homeAway = []
        repeat = []
        robin1 = []

        for i in range(5):
            file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/' + str(i) + '/Violations' + str(j) + '.txt', "r")
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway.extend(text[0])
            repeat.extend(text[1])
            robin1.extend(text[2])

        total = []
        for i in range(len(homeAway)):
            total.append(homeAway[i]+repeat[i]+robin1[i])
        total = np.array(total)

        weights = np.ones_like(total)/float(len(total))
        hist, bins, _ = plt.hist(total, bins=30, weights=weights, label='Total', color='Green', edgecolor='black', linewidth=1.2)

        n = len(hist)
        x_hist=np.zeros((n),dtype=float) 
        for ii in range(n):
            x_hist[ii]=(bins[ii+1]+bins[ii])/2

        y_hist=hist
        mean = sum(x_hist*y_hist)/sum(y_hist)                  
        sigma = np.std(total)

        popt, pcov = curve_fit(gaus, x_hist, y_hist, p0=[max(y_hist),mean,sigma])
        x_hist_2=np.linspace(np.min(0), np.max(x_hist)+np.min(x_hist), np.min(x_hist)+np.max(x_hist))

        cdf = np.cumsum(gaus(x_hist_2, *popt))
        cdf = cdf/np.max(cdf)

        chance = np.interp(0.5, x_hist_2, cdf)

        totalChance.append(chance)

    file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Chances.txt', 'w')
    np.savetxt(file, [totalChance], delimiter=',')
    file.close()

def plotParameters():
    times = []
    for i in range(4, 51, 2):
        current = []
        for j in range(5):
            file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/' + str(j) + '/time' + str(i) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            current.append(text)

        times.append(np.mean(current)/1000000)

    aHA, bHA, interceptHA, aRepeat, bRepeat, interceptRepeat, aRobin, bRobin, interceptRobin, aTotal, bTotal, interceptTotal, labels = ([] for i in range(13))
    for i in range(4, 51, 2):
        file = open('E:/NewExperiment/NewResults/MinViolationExperiment2/Parameters3/Parameters' + str(i) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        labels.append(i)

        aHA.append(text[0])
        bHA.append(text[1])
        interceptHA.append(text[2])
        aRepeat.append(text[3])
        bRepeat.append(text[4])
        interceptRepeat.append(text[5])
        aRobin.append(text[6])
        bRobin.append(text[7])
        interceptRobin.append(text[8])
        aTotal.append(text[9])
        bTotal.append(text[10])
        interceptTotal.append(text[11])

    plt.rcParams.update({'font.size': 25})

    plt.plot(labels, aTotal, color='Green', label='Total')
    plt.plot(labels, aRobin, color='Red', label='Double Round-Robin')
    plt.plot(labels, aHA, color='Blue', label='maxStreak')
    plt.plot(labels, aRepeat, color='Orange', label='noRepeat')

    fitaHA = np.poly1d(np.polyfit(labels, aHA, 2))
    MSEHA = np.square(np.subtract(aHA, fitaHA(labels))).mean()
    rmseHA = math.sqrt(MSEHA)
    #print(rmseHA)

    fitaRepeat = np.poly1d(np.polyfit(labels, aRepeat, 2))
    MSERepeat  = np.square(np.subtract(aRepeat, fitaRepeat(labels))).mean()
    rmseRepeat  = math.sqrt(MSERepeat)
    #print(rmseRepeat)

    fitaRobin = np.poly1d(np.polyfit(labels, aRobin, 2))
    MSERobin  = np.square(np.subtract(aRobin, fitaRobin(labels))).mean()
    rmseRobin  = math.sqrt(MSERobin)
    #print(rmseRobin)

    fitaTotal = np.poly1d(np.polyfit(labels, aTotal, 2))
    MSETotal  = np.square(np.subtract(aTotal, fitaTotal(labels))).mean()
    rmseTotal  = math.sqrt(MSETotal)
    #print(rmseTotal)

    coefHA = []
    for i in fitaHA.coefficients:
        coefHA.append(round(i, 3))

    coefRepeat = []
    for i in fitaRepeat.coefficients:
        coefRepeat.append(round(i, 3))

    coefRobin = []
    for i in fitaRobin.coefficients:
        coefRobin.append(round(i, 3))

    coefTotal = []
    for i in fitaTotal.coefficients:
        coefTotal.append(round(i, 3))

    a = str(coefTotal[0]) + 'x$^2$ - ' + str(coefTotal[1])[1:] + 'x - ' + str(coefTotal[2])[1:]
    b = str(coefRobin[0]) + 'x$^2$ - ' + str(coefRobin[1])[1:] + 'x + ' + str(coefRobin[2])
    c = str(coefHA[0]) + 'x$^2$ + ' + str(coefHA[1]) + 'x - ' + str(coefHA[2])[1:]
    d = str(coefRepeat[0]) + 'x$^2$ + ' + str(coefRepeat[1]) + 'x + ' + str(coefRepeat[2])

    plt.plot(labels, fitaTotal(labels), color='Green', ls='--', label=a)
    plt.plot(labels, fitaRobin(labels), color='Red', ls='--', label=b)
    plt.plot(labels, fitaHA(labels), color='Blue', ls='--', label=c)
    plt.plot(labels, fitaRepeat(labels), color='Orange', ls='--', label=d)

    plt.xlabel('Number of Teams')
    plt.ylabel('y-intercept (a)')
    plt.legend(loc='upper left')
    plt.show()

    fitbHA = np.poly1d(np.polyfit(labels, bHA, 1))
    MSEHA = np.square(np.subtract(bHA, fitbHA(labels))).mean()
    rmseHA = math.sqrt(MSEHA)
    #print(rmseHA)

    fitbRepeat = np.poly1d(np.polyfit(labels, bRepeat, 1))
    MSERepeat  = np.square(np.subtract(bRepeat, fitbRepeat(labels))).mean()
    rmseRepeat  = math.sqrt(MSERepeat)
    #print(rmseRepeat)

    fitbRobin = np.poly1d(np.polyfit(labels, bRobin, 1))
    MSERobin  = np.square(np.subtract(bRobin, fitbRobin(labels))).mean()
    rmseRobin  = math.sqrt(MSERobin)
    #print(rmseRobin)

    fitbTotal = np.poly1d(np.polyfit(labels, bTotal, 1))
    MSETotal  = np.square(np.subtract(bTotal, fitbTotal(labels))).mean()
    rmseTotal  = math.sqrt(MSETotal)
    #print(rmseTotal)

    coefHA = []
    for i in fitbHA.coefficients:
        coefHA.append(round(i, 3))

    coefRepeat = []
    for i in fitbRepeat.coefficients:
        coefRepeat.append(round(i, 3))

    coefRobin = []
    for i in fitbRobin.coefficients:
        coefRobin.append(round(i, 3))

    coefTotal = []
    for i in fitbTotal.coefficients:
        coefTotal.append(round(i, 3))
    
    
    a = str(coefHA[0]) + 'x - ' + str(coefHA[1])[1:]
    b = str(coefRepeat[0]) + 'x ' + str(coefRepeat[1])
    c = str(coefRobin[0]) + 'x ' + str(coefRobin[1])
    d = str(coefTotal[0]) + 'x ' + str(coefTotal[1])

    plt.plot(labels, bTotal, color='Green', label='Total')
    plt.plot(labels, bRobin, color='Red', label='Double Round-Robin')
    plt.plot(labels, bHA, color='Blue', label='maxStreak')
    plt.plot(labels, bRepeat, color='Orange', label='noRepeat')

    plt.plot(labels, fitbTotal(labels), color='Green', ls='--', label=a)
    plt.plot(labels, fitbRobin(labels), color='Red', ls='--', label=b)
    plt.plot(labels, fitbHA(labels), color='Blue', ls='--', label=c)
    plt.plot(labels, fitbRepeat(labels), color='Orange', ls='--', label=d)

    plt.xlabel('Number of Teams')
    plt.ylabel('b')
    plt.legend(loc='lower left')

    plt.show()


    #plt.plot(labels, np.log(interceptHA), color='Blue', label='maxStreak')
    #plt.plot(labels, np.log(interceptRepeat), color='Orange', label='noRepeat')
    #plt.plot(labels, np.log(interceptRobin), color='Red', label='Double Round-Robin')
    #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    #fig.add_subplot(111, frameon=False)
    #plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    #print(len(interceptTotal))

    fig, ax1 = plt.subplots()

    one = ax1.plot(labels, np.log(interceptTotal), color='Green', label='Total')

    fit = np.polyfit(x=labels, y=np.log(interceptTotal), deg=1, full=True)
    fitaHA = np.poly1d(fit[0])

    #print(fitaHA)
    #print(fit[1])
    MSEHA = np.square(np.subtract(np.log(interceptTotal), fitaHA(labels))).mean()
    rmseHA = math.sqrt(MSEHA)
    #print(rmseHA)

    coefTotal = []
    for i in fitaHA.coefficients:
        coefTotal.append(round(i, 3))

    a = str(coefTotal[0]) + 'x - ' + str(coefTotal[1])[1:]
    
    timeNeeded = []
    for i in range(24):
        timeNeeded.append(times[i]*interceptTotal[i])

    print(interceptTotal[0])
    print(times[0])
    print(timeNeeded[0])

    #print(np.exp(fitaHA(labels)))

    two = ax1.plot(labels, (fitaHA(labels)), color='Green', ls='--', label=a)

    ax2 = ax1.twinx()
    ax2.set_ylabel('x-intercept ($10^{138}$)')
    ax2.plot(labels, interceptTotal, ls='None')

    ax1.set_xlabel('Number of Teams')
    ax1.set_ylabel('ln(x-intercept)')

    test = fig.legend(['Total', a], loc='upper center', ncols=2, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    ax2.yaxis.offsetText.set_visible(False)

    plt.show()

    exit()

def finalGraphs():
    for i in range(6, 61, 6):
        print(i)
        directory = 'E:/NewExperiment/NewResults/PPA/' + str(i)
        for j in range(11):
            file = open(directory + '/Violations' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway = text[0].astype(int)
            repeat = text[1].astype(int)
            robin = text[2].astype(int)
            evals = text[5].astype(int)
            total = []

            # for z in range(len(homeAway)):
            #     if z != 0:
            #         if homeAway[z-1] <= homeAway[z]:
            #             homeAway[z] = homeAway[z-1]
            #         if repeat[z-1] <= repeat[z]:
            #             repeat[z] = repeat[z-1]
            #         if robin[z-1] <= robin[z]:
            #             robin[z] = robin[z-1]

            newHomeAway = []
            newRepeat = []
            newRobin = []

            # counter = 0
            # last = 0
            # for z in evals:
            #     for x in range(z-last):
            #         newHomeAway.append(homeAway[counter])
            #         newRepeat.append(repeat[counter])
            #         newRobin.append(robin[counter])
            #     last = z
            #     counter += 1
            
            # homeAway = newHomeAway
            # repeat = newRepeat
            # robin = newRobin
            
            for z in range(len(homeAway)):
                total.append(homeAway[z] + repeat[z] + robin[z])

            plt.rcParams.update({'font.size': 25})

            plt.plot(total, label='Total', color='Green')
            plt.plot(robin, label='Double Round-Robin', color='Red')
            plt.plot(homeAway, label='maxStreak', color='Blue')
            plt.plot(repeat, label='noRepeat', color='Orange')

            plt.legend(loc='upper right')

            plt.xlabel('Evaluations')
            plt.ylabel('Number of Violations')
            
            figure = plt.gcf()
            figure.set_size_inches(19.2, 10.8)

            plt.savefig(directory + '/Graphs/' + 'PPAGens' + str(j) + '.png')

            plt.clf()

def summarize():
    values = []
    left = []
    evals = []

    for i in range(6, 61, 6):
        directory = 'E:/NewExperiment/NewResults/PPA/' + str(i)
        counter = 0
        currLeft = []
        currEvals = []
        for j in range(11):
            file = open(directory + '/Violations' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway = text[0].astype(int)
            repeat = text[1].astype(int)
            robin = text[2].astype(int)
            evals2 = text[5].astype(int)
            total = []

            newHomeAway = []
            newRepeat = []
            newRobin = []

            counter2 = 0
            last = 0
            for z in evals2:
                for x in range(z-last):
                    newHomeAway.append(homeAway[counter2])
                    newRepeat.append(repeat[counter2])
                    newRobin.append(robin[counter2])
                last = z
                counter2 += 1
            
            homeAway = newHomeAway
            repeat = newRepeat
            robin = newRobin

            for z in range(len(homeAway)):
                total.append(homeAway[z] + repeat[z] + robin[z])

            if len(total) < 1000000:
                counter += 1
                currEvals.append(len(total))
            else:
                currLeft.append(total[len(total)-1])


        if len(currLeft) > 0:
            left.append(np.mean(currLeft))
        else:
            left.append(0)

        if len(currEvals) > 0:
            evals.append(np.mean(currEvals))
        else:
            evals.append(1000000)

        values.append(counter)

    print(values)

    print()

    print(left)

    print()
    
    print(evals)

    file = open('E:/NewExperiment/NewResults/PPA/Summary.txt', 'w')
    np.savetxt(file, [values, left, evals], delimiter=',')
    file.close()

def getValidRemaining():
    x = []
    for i in range(6, 61, 6):
        x.append(i)

    directory = 'E:/NewExperiment/NewResults/'
    valid = []
    left = []
    evals = []
    file = open(directory + 'PPA/Summary.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    valid.append(text[0])
    left.append(text[1])
    evals.append(text[2])
    file = open(directory + 'SASC/Summary.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    valid.append(text[0])
    left.append(text[1])
    evals.append(text[2])
    file = open(directory + 'SALR/Summary.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    valid.append(text[0])
    left.append(text[1])
    evals.append(text[2])
    file = open(directory + 'SAGG/Summary.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    valid.append(text[0])
    left.append(text[1])
    evals.append(text[2])
    file = open(directory + 'HC/Summary.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    valid.append(text[0])
    left.append(text[1])
    evals.append(text[2])

    plt.rcParams.update({'font.size': 25})

    fig, ax1 = plt.subplots(1, 1)

    #fig.add_subplot(111, frameon=False)
    #plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    #ax2.yaxis.tick_right()

    e = plt.plot(x, left[4], label='HC', color='orange')
    b = plt.plot(x, left[1], label='SASC', color='purple')
    c = plt.plot(x, left[2], label='SALR', color='green')
    d = plt.plot(x, left[3], label='SAGG', color='blue')
    a = plt.plot(x, left[0], label='PPA', color='red')

    test = fig.legend(['HC', 'SASC', 'SALR', 'SAGG', 'PPA'], loc='upper center', ncols=5, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    plt.xlabel('Number of Teams')
    plt.ylabel('Violations Remaining')

    plt.show()
    
    evalsHC = []
    evalsSASC = []
    evalsSALR = []
    evalsSAGG = []
    evalsPPA = []

    evalsHC.append(975530)
    evalsHC.append(1631693)
    evalsHC.append(1957336)
    evalsHC.append(2071230)
    evalsHC.append(2819571)
    xHC = [36, 42, 48, 54, 60]

    evalsSASC.append(938550)
    evalsSASC.append(1271117)
    evalsSASC.append(1499089)
    evalsSASC.append(1854768)
    evalsSASC.append(2191180)
    evalsSASC.append(2715322)
    evalsSASC.append(3017446)
    evalsSASC.append(3475134)
    xSASC = [18, 24, 30, 36, 42, 48, 54, 60]

    evalsSALR.append(753472)
    evalsSALR.append(1123317)
    evalsSALR.append(1456800)
    evalsSALR.append(1615549)
    evalsSALR.append(1695022)
    evalsSALR.append(1892532)
    xSALR = [30, 36, 42, 48, 54, 60]

    evalsSAGG.append(886697)
    evalsSAGG.append(1607192)
    evalsSAGG.append(1775653)
    evalsSAGG.append(2227969)
    evalsSAGG.append(2454213)
    xSAGG = [36, 42, 48, 54, 60]

    evalsPPA.append(353307)
    evalsPPA.append(1448804)
    evalsPPA.append(1704709)
    evalsPPA.append(2410992)
    evalsPPA.append(2704579)
    evalsPPA.append(3356063)
    evalsPPA.append(4917093)
    evalsPPA.append(6275576)
    xPPA = [18, 24, 30, 36, 42, 48, 54, 60]

    xHC2 = [6, 12, 18, 24, 30, 36]
    evalsHC2 = evals[4][:len(xHC2)]

    xSASC2 = [6, 12, 18]
    evalsSASC2 = evals[1][:len(xSASC2)]

    xSALR2 = [6, 12, 18, 24, 30]
    evalsSALR2 = evals[2][:len(xSALR2)]

    xSAGG2 = [6, 12, 18, 24, 30, 36]
    evalsSAGG2 = evals[3][:len(xSAGG2)]

    xPPA2 = [6, 12, 18]
    evalsPPA2 = evals[0][:len(xPPA2)]

    fig, ax = plt.subplots()

    j = plt.plot(xHC2, evalsHC2, label='HC', color='orange')
    g = plt.plot(xSASC2, evalsSASC2, label='SASC', color='purple')
    h = plt.plot(xSALR2, evalsSALR2, label='SALR', color='green')
    i = plt.plot(xSAGG2, evalsSAGG2, label='SAGG', color='blue')
    f = plt.plot(xPPA2, evalsPPA2, label='PPA', color='red')

    plt.plot(xHC, evalsHC, label='HC', ls='--', dashes=(5, 5), color=j[0].get_color())
    plt.plot(xSASC, evalsSASC, label='HC', ls='--', dashes=(5, 5), color=g[0].get_color())
    plt.plot(xSALR, evalsSALR, label='HC', ls='--', dashes=(5, 5), color=h[0].get_color())
    plt.plot(xSAGG, evalsSAGG, label='HC', ls='--', dashes=(5, 5), color=i[0].get_color())
    plt.plot(xPPA, evalsPPA, label='HC', ls='--', dashes=(5, 5), color=f[0].get_color())

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,1))
    ax.yaxis.offsetText.set_visible(False)

    test = fig.legend(['HC', 'SASC', 'SALR', 'SAGG', 'PPA'], loc='upper center', ncols=5, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    plt.xlabel('Number of Teams')
    plt.ylabel('Evaluations ($10^6$)')

    #ax1.legend(['PPA', 'SASC', 'SALR', 'SAGG', 'HC', a, b, c, d, e])
    #ax2.legend(['PPA', 'SASC', 'SALR', 'SAGG', 'HC', f, g, h, i, j])

    #ax2.set_ylabel('Evaluations', rotation=90)
    #ax2.yaxis.set_label_position("right")

    plt.show()

    file = open('E:/NewExperiment/NewResults/HC/SummaryTimePer.txt', 'r')
    timeHC = np.loadtxt(file, delimiter=',')
    file.close()

    file = open('E:/NewExperiment/NewResults/SASC/SummaryTimePer.txt', 'r')
    timeSASC = np.loadtxt(file, delimiter=',')
    file.close()

    file = open('E:/NewExperiment/NewResults/SALR/SummaryTimePer.txt', 'r')
    timeSALR = np.loadtxt(file, delimiter=',')
    file.close()

    file = open('E:/NewExperiment/NewResults/SAGG/SummaryTimePer.txt', 'r')
    timeSAGG = np.loadtxt(file, delimiter=',')
    file.close()

    file = open('E:/NewExperiment/NewResults/PPA/SummaryTimePer.txt', 'r')
    timePPA = np.loadtxt(file, delimiter=',')
    file.close()

    def calc(arr, x):
        result = []
        for i in range(len(x)):
            result.append(arr[i]*x[i])
        return result
    
    timesHC = calc(timeHC, evalsHC2)
    timesSASC = calc(timeSASC, evalsSASC2)
    timesSALR = calc(timeSALR, evalsSALR2)
    timesSAGG = calc(timeSAGG, evalsSAGG2)
    timesPPA = calc(timePPA, evalsPPA2)
    timesSAGG[0] = 0.42

    timesHC2 = calc(timeHC[10-len(evalsHC):10], evalsHC)
    timesSASC2 = calc(timeSASC[10-len(evalsSASC):10], evalsSASC)
    timesSALR2 = calc(timeSALR[10-len(evalsSALR):10], evalsSALR)
    timesSAGG2 = calc(timeSAGG[10-len(evalsSAGG):10], evalsSAGG)
    timesPPA2 = calc(timePPA[10-len(evalsPPA):10], evalsPPA)

    print(timesHC + timesHC2)
    print(timesSASC + timesSASC2)
    print(timesSALR + timesSALR2)
    print(timesSAGG + timesSAGG2)
    print(timesPPA + timesPPA2)

    fig, ax = plt.subplots()

    j = plt.plot(xHC2, timesHC, label='HC', color='orange')
    g = plt.plot(xSASC2, timesSASC, label='SASC', color='purple')
    h = plt.plot(xSALR2, timesSALR, label='SALR', color='green')
    i = plt.plot(xSAGG2, timesSAGG, label='SAGG', color='blue')
    f = plt.plot(xPPA2, timesPPA, label='PPA', color='red')

    plt.plot(xHC, timesHC2, label='HC', ls='--', dashes=(5, 5), color=j[0].get_color())
    plt.plot(xSASC, timesSASC2, label='HC', ls='--', dashes=(5, 5), color=g[0].get_color())
    plt.plot(xSALR, timesSALR2, label='HC', ls='--', dashes=(5, 5), color=h[0].get_color())
    plt.plot(xSAGG, timesSAGG2, label='HC', ls='--', dashes=(5, 5), color=i[0].get_color())
    plt.plot(xPPA, timesPPA2, label='HC', ls='--', dashes=(5, 5), color=f[0].get_color())

    test = fig.legend(['HC', 'SASC', 'SALR', 'SAGG', 'PPA'], loc='upper center', ncols=5, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    plt.xlabel('Number of Teams')
    plt.ylabel('Time(sec)')

    plt.show()

def getTime():
    x = []
    for i in range(4, 51, 2):
        x.append(i)
    x = np.array(x)
    times = []
    for i in range(4, 51, 2):
        current = []
        for j in range(5):
            file = open('E:/NewExperiment/NewResults/MinViolationExperiment/' + str(j) + '/time' + str(i) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            current.append(text)

        times.append(np.mean(current)/1000000)

    fitTime = np.poly1d(np.polyfit(x, times, 2))

    mseTime = np.square(np.subtract(times, fitTime(x))).mean()
    rmseTime = math.sqrt(mseTime)
    print(rmseTime)

    print(fitTime)

    coefTime = []
    for i in fitTime.coefficients:
        coefTime.append(round(i, 3))

    a = '(' + str(fitTime.coefficients[0])[:5] + 'x$10^{-6}$)x$^2$ - (' + str(fitTime.coefficients[1])[1:6] + 'x$10^{-5}$)x + ' + '0.358x$10^{-4}$'

    plt.rcParams.update({'font.size': 25})

    plt.xlabel('Number of teams')
    plt.ylabel('Time(sec)')

    plt.plot(x, times, label='Time taken')
    plt.plot(x, fitTime(x), ls='--', label=a)
    plt.legend(loc='best')

    plt.show()

def getTimeAlgs():
    times = []
    timesValid = []
    timesInvalid = []

    for i in range(6, 61, 6):
        directory = 'E:/NewExperiment/NewResults/PPA/' + str(i)
        currTimes = []
        currTimesValid = []
        currTimesInvalid = []
        for j in range(11):
            file = open(directory + '/Times' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            time = text
            currTimes.append(time)

            file = open(directory + '/Violations' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            if len(text[0]) < 1000000:
                currTimesValid.append(time)
            else:
                currTimesInvalid.append(time)

        if len(currTimesValid) > 0:
            timesValid.append(np.mean(currTimesValid))
        else:
            timesValid.append(0)

        if len(currTimesInvalid) > 0:
            timesInvalid.append(np.mean(currTimesInvalid))
        else:
            timesInvalid.append(0)

        times.append(np.mean(currTimes))

    file = open('E:/NewExperiment/NewResults/PPA/SummaryTime.txt', 'w')
    np.savetxt(file, [timesValid, timesInvalid, times], delimiter=',')
    file.close()

def getTimePerAlgs():
    times = []

    for i in range(6, 61, 6):
        directory = 'E:/NewExperiment/NewResults/PPA/' + str(i)
        currTimes = []
        for j in range(11):
            file = open(directory + '/Times' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            time = text

            file = open(directory + '/Violations' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            total = 0
            for z in range(len(text[5])):
                if z != 0:
                    total += text[5][z] - total
                else:
                    total += text[5][z]


            currTimes.append(time/total)

        times.append(np.mean(currTimes))

    file = open('E:/NewExperiment/NewResults/PPA/SummaryTimePer.txt', 'w')
    np.savetxt(file, [times], delimiter=',')
    file.close()

def makeTimeAlgs():
    x = []
    for i in range(6, 61, 6):
        x.append(i)

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'PPA/SummaryTimePer.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    ppaPer = (text)
    file = open(directory + 'SASC/SummaryTimePer.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    sascPer = text
    file = open(directory + 'SALR/SummaryTimePer.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    salrPer = text
    file = open(directory + 'SAGG/SummaryTimePer.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    saggPer = text
    file = open(directory + 'HC/SummaryTimePer.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    hcPer = text

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'PPA/SummaryTime.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    ppa = text[2]
    file = open(directory + 'SASC/SummaryTime.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    sasc = text[2]
    file = open(directory + 'SALR/SummaryTime.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    salr = text[2]
    file = open(directory + 'SAGG/SummaryTime.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    sagg = text[2]
    file = open(directory + 'HC/SummaryTime.txt', 'r')
    text = np.loadtxt(file, delimiter=',')
    file.close()
    hc = text[2]


    plt.rcParams.update({'font.size': 25})

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    ax1.xaxis.offsetText.set_visible(False)
    ax2.xaxis.offsetText.set_visible(False)

    ax2.yaxis.tick_right()
    
    a = ax1.plot(x, ppa, label='PPA')
    b = ax1.plot(x, hc, label='HC')
    c = ax1.plot(x, sasc, label='SASC')
    d = ax1.plot(x, salr, label='SALR')
    e = ax1.plot(x, sagg, label='SAGG')

    f = ax2.plot(x, ppaPer, label='PPA')
    g = ax2.plot(x, hcPer, label='PPA')
    h = ax2.plot(x, sascPer, label='PPA')
    i = ax2.plot(x, salrPer, label='PPA')
    j = ax2.plot(x, saggPer, label='PPA')

    ax1.legend(['PPA', 'HC', 'SASC', 'SALR', 'SAGG', a, b, c, d, e])
    ax2.legend(['PPA', 'HC', 'SASC', 'SALR', 'SAGG', f, g, h, i, j])

    plt.xlabel('Number of teams')
    ax1.set_ylabel('Time(sec)')

    plt.show()

def getBest():
    for i in range(6, 61, 6):
        directory = 'E:/NewExperiment/NewResults/HC/'+ str(i)
        size = []
        final = []
        for j in range(11):
            file = open(directory + '/Violations' + str(j) + '.txt', 'r')
            text = np.loadtxt(file, delimiter=',')
            file.close()

            homeAway = text[0].astype(int)
            repeat = text[1].astype(int)
            robin = text[2].astype(int)
            #evals2 = text[5].astype(int)
            total = []

            #newHomeAway = []
            #newRepeat = []
            #newRobin = []

            #counter2 = 0
            #last = 0
            #for z in evals2:
            #    for x in range(z-last):
            #        newHomeAway.append(homeAway[counter2])
            #        newRepeat.append(repeat[counter2])
            #        newRobin.append(robin[counter2])
            #    last = z
            #    counter2 += 1

            #homeAway = newHomeAway
            #repeat = newRepeat
            #robin = newRobin

            for z in range(len(homeAway)):
                total.append(homeAway[z] + repeat[z] + robin[z])
            
            final.append(total[len(total)-1])
            size.append(len(total))

        if np.count_nonzero(final) == 11:
            best = final.index(np.min(final))
            median = final.index(np.median(final))
        elif np.count_nonzero(final) > 5:
            best = size.index(np.min(size))
            median = final.index(np.median(final))
        else:
            best = size.index(np.min(size))
            median = size.index(np.median(size))
            
        file = open('E:/NewExperiment/NewResults/HC/' + str(i) + '/BestMed.txt', 'w')
        np.savetxt(file, [best, median], fmt='%i', delimiter=',')
        file.close()

def bestGraphs(j):
    for i in range(6, 61, 6):
        file = open('E:/NewExperiment/NewResults/HC/' + str(j) + '/BestMed.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        bestHC = text[0].astype(int)
        medHC = text[1].astype(int)

        file = open('E:/NewExperiment/NewResults/SASC/' + str(j) + '/BestMed.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        bestSASC = text[0].astype(int)
        medSASC = text[1].astype(int)

        file = open('E:/NewExperiment/NewResults/SALR/' + str(j) + '/BestMed.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        bestSALR = text[0].astype(int)
        medSALR = text[1].astype(int)

        file = open('E:/NewExperiment/NewResults/SAGG/' + str(j) + '/BestMed.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        bestSAGG = text[0].astype(int)
        medSAGG = text[1].astype(int)

        file = open('E:/NewExperiment/NewResults/PPA/' + str(j) + '/BestMed.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        bestPPA = text[0].astype(int)
        medPPA = text[1].astype(int)

        file = open('E:/NewExperiment/NewResults/HC/' + str(j) + '/Violations' + str(bestHC) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAwayHC = text[0].astype(int)
        repeatHC = text[1].astype(int)
        robinHC = text[2].astype(int)

        totalHC = []
        for z in range(len(homeAwayHC)):
            totalHC.append(homeAwayHC[z] + repeatHC[z] + robinHC[z])

        file = open('E:/NewExperiment/NewResults/SASC/' + str(j) + '/BestViolations' + str(bestSASC) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAwaySASC = text[0].astype(int)
        repeatSASC = text[1].astype(int)
        robinSASC = text[2].astype(int)

        totalSASC = []
        for z in range(len(homeAwaySASC)):
            totalSASC.append(homeAwaySASC[z] + repeatSASC[z] + robinSASC[z])

        file = open('E:/NewExperiment/NewResults/SALR/' + str(j) + '/BestViolations' + str(bestSALR) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAwaySALR = text[0].astype(int)
        repeatSALR = text[1].astype(int)
        robinSALR = text[2].astype(int)

        totalSALR = []
        for z in range(len(homeAwaySALR)):
            totalSALR.append(homeAwaySALR[z] + repeatSALR[z] + robinSALR[z])

        file = open('E:/NewExperiment/NewResults/SAGG/' + str(j) + '/BestViolations' + str(bestSAGG) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAwaySAGG = text[0].astype(int)
        repeatSAGG = text[1].astype(int)
        robinSAGG = text[2].astype(int)

        totalSAGG = []
        for z in range(len(homeAwaySAGG)):
            totalSAGG.append(homeAwaySAGG[z] + repeatSAGG[z] + robinSAGG[z])

        file = open('E:/NewExperiment/NewResults/PPA/' + str(j) + '/Violations' + str(bestPPA) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAwayPPA = text[0].astype(int)
        repeatPPA  = text[1].astype(int)
        robinPPA  = text[2].astype(int)
        evals2 = text[5].astype(int)
        totalPPA = []

        newHomeAway = []
        newRepeat = []
        newRobin = []

        counter2 = 0
        last = 0
        for z in evals2:
            for x in range(z-last):
                newHomeAway.append(homeAwayPPA[counter2])
                newRepeat.append(repeatPPA[counter2])
                newRobin.append(robinPPA[counter2])
            last = z
            counter2 += 1
            
        homeAwayPPA = newHomeAway
        repeatPPA = newRepeat
        robinPPA = newRobin

        for z in range(len(homeAwayPPA)):
            totalPPA.append(homeAwayPPA[z] + repeatPPA[z] + robinPPA[z])

        return [totalPPA, totalHC, totalSASC, totalSALR, totalSAGG], [np.log1p(totalPPA), np.log1p(totalHC), np.log1p(totalSASC), np.log1p(totalSALR), np.log1p(totalSAGG)]

        plt.rcParams.update({'font.size': 25})

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax2.yaxis.tick_right()
        
        ax1.plot(totalPPA, label='PPA')
        ax1.plot(totalHC, label='HC')
        ax1.plot(totalSASC, label='SASC')
        ax1.plot(totalSALR, label='SALR')
        ax1.plot(totalSAGG, label='SAGG')

        ax2.plot(np.log1p(totalPPA), label='PPA')
        ax2.plot(np.log1p(totalHC), label='HC')
        ax2.plot(np.log1p(totalSASC), label='SASC')
        ax2.plot(np.log1p(totalSALR), label='SALR')
        ax2.plot(np.log1p(totalSAGG), label='SAGG')

        ax1.legend(loc='best')

        plt.subplots_adjust(wspace=0.042)
        plt.xlabel('Evaluations')
        ax1.set_ylabel('Number of Violations')
        ax2.set_ylabel('ln(Number of Violations)', rotation=90)
        ax2.yaxis.set_label_position("right")

        print(i)

        plt.show()

        exit()

def makeBestGraphs():
    total6, logTotal6 = bestGraphs(12)
    total30, logTotal30 = bestGraphs(30)
    total60, logTotal60 = bestGraphs(60)

    plt.rcParams.update({'font.size': 20})

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax2.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax6.yaxis.tick_right()

    ax1.plot(total6[1], label='HC', color='orange')
    ax1.plot(total6[2], label='SASC', color='purple')
    ax1.plot(total6[3], label='SALR', color='green')
    ax1.plot(total6[4], label='SAGG', color='blue')
    ax1.plot(total6[0], label='PPA', color='red')

    ax2.plot(logTotal6[1], label='HC', color='orange')
    ax2.plot(logTotal6[2], label='SASC', color='purple')
    ax2.plot(logTotal6[3], label='SALR', color='green')
    ax2.plot(logTotal6[4], label='SAGG', color='blue')
    ax2.plot(logTotal6[0], label='PPA', color='red')

    ax3.plot(total30[1], label='HC', color='orange')
    ax3.plot(total30[2], label='SASC', color='purple')
    ax3.plot(total30[3], label='SALR', color='green')
    ax3.plot(total30[4], label='SAGG', color='blue')
    ax3.plot(total30[0], label='PPA', color='red')

    ax4.plot(logTotal30[1], label='HC', color='orange')
    ax4.plot(logTotal30[2], label='SASC', color='purple')
    ax4.plot(logTotal30[3], label='SALR', color='green')
    ax4.plot(logTotal30[4], label='SAGG', color='blue')
    ax4.plot(logTotal30[0], label='PPA', color='red')

    ax5.plot(total60[1], label='HC', color='orange')
    ax5.plot(total60[2], label='SASC', color='purple')
    ax5.plot(total60[3], label='SALR', color='green')
    ax5.plot(total60[4], label='SAGG', color='blue')
    ax5.plot(total60[0], label='PPA', color='red')

    ax6.plot(logTotal60[1], label='HC', color='orange')
    ax6.plot(logTotal60[2], label='SASC', color='purple')
    ax6.plot(logTotal60[3], label='SALR', color='green')
    ax6.plot(logTotal60[4], label='SAGG', color='blue')
    ax6.plot(logTotal60[0], label='PPA', color='red')

    plt.subplots_adjust(hspace=0.2, wspace=0.042, top=0.95)

    fig.text(0.065, 0.5, 'Number of Violations', va='center', rotation='vertical', fontsize=20)
    fig.text(0.93, 0.5, 'ln(Number of Violations)', va='center', rotation=270, fontsize=20)
    fig.text(0.455, 0.07, 'Evaluations ($10^6$)', fontsize=20)

    ax1.ticklabel_format(style='sci', axis='x', scilimits=(6,6))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(6,6))

    ax1.text(0.5, 0.92, '12 Teams', fontweight='bold', horizontalalignment='center', transform=ax1.transAxes)
    ax2.text(0.5, 0.92, '12 Teams-ln', fontweight='bold', horizontalalignment='center', transform=ax2.transAxes)
    ax3.text(0.5, 0.92, '30 Teams', fontweight='bold', horizontalalignment='center', transform=ax3.transAxes)
    ax4.text(0.5, 0.92, '30 Teams-ln', fontweight='bold', horizontalalignment='center', transform=ax4.transAxes)
    ax5.text(0.5, 0.92, '60 Teams', fontweight='bold', horizontalalignment='center', transform=ax5.transAxes)
    ax6.text(0.5, 0.92, '60 Teams-ln', fontweight='bold', horizontalalignment='center', transform=ax6.transAxes)

    test = fig.legend(['HC', 'SASC', 'SALR', 'SAGG', 'PPA'], loc='upper center', ncols=5, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    ax1.xaxis.offsetText.set_visible(False)
    ax2.xaxis.offsetText.set_visible(False)
    ax3.xaxis.offsetText.set_visible(False)
    ax4.xaxis.offsetText.set_visible(False)
    ax5.xaxis.offsetText.set_visible(False)
    ax6.xaxis.offsetText.set_visible(False)

    fig.set_size_inches(19.2, 20)

    fig.savefig('C:/Users/megki_000/Desktop/BestGraphs.png')

    plt.show()

def tempGraphs():
    temperatureLR = []
    originalTemp = 3.321928094887363
    temp = originalTemp
    temperatureLR.append(originalTemp)
    phase = 0
    while phase < 10:
        counter = 0
        while counter != 100000:
            counter += 1
            temp = temp-(originalTemp/100000)
            temperatureLR.append(temp)
        phase += 1
        temp = originalTemp/2
        originalTemp = temp

    temperatureGG = []
    counter = 1
    temp = 1/math.log(counter+1, 10)
    temperatureGG.append(temp)

    while counter != 1000001:
        counter += 1
        temp = 1/math.log(counter+1, 10)
        temperatureGG.append(temp)

    temperatureSC = []
    originalTemp = 3.32
    temp = originalTemp
    counter = 0
    phase = 0
    temperatureSC.append(originalTemp)

    while phase < 10:
        counter = 0
        while counter != 100000:
            counter += 1
            temperatureSC.append(temp)
        phase += 1
        temp = temp-(originalTemp/10)

    plt.rcParams.update({'font.size': 20})

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    ax0.plot(temperatureGG, color='Blue')
    ax1.plot(temperatureSC, color='Blue')
    ax2.plot(temperatureLR, color='Blue')

    ax0.xaxis.offsetText.set_visible(False)
    ax1.xaxis.offsetText.set_visible(False)
    ax2.xaxis.offsetText.set_visible(False)

    ax0.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)

    ax1.yaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0.02, bottom=0.35)
    
    ax0.text(0.15, 0.95, 'Geman & Geman (c=1)', fontweight='bold', horizontalalignment='left', transform=ax0.transAxes)
    ax1.text(0.35, 0.95, 'Staircase', fontweight='bold', horizontalalignment='left', transform=ax1.transAxes)
    ax2.text(0.28, 0.95, 'Linear Reheat', fontweight='bold', horizontalalignment='left', transform=ax2.transAxes)

    ax0.set_ylabel('Temperature')
    ax1.set_xlabel('Evaluations ($10^6$)')

    plt.show()

def HCPPAGraph():
    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'HC/18/Violations0.txt', 'r')
    HC = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwayHC = HC[0].astype(int)
    repeatHC = HC[1].astype(int)
    robinHC = HC[2].astype(int)

    totalHC = []

    for z in range(len(homeAwayHC)):
        totalHC.append(homeAwayHC[z] + repeatHC[z] + robinHC[z])

    file = open(directory + 'PPA/18/Violations9.txt', 'r')
    PPA = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwayPPA = PPA[0].astype(int)
    repeatPPA  = PPA[1].astype(int)
    robinPPA  = PPA[2].astype(int)
    evals2 = PPA[5].astype(int)
    totalPPA = []

    newHomeAway = []
    newRepeat = []
    newRobin = []

    counter2 = 0
    last = 0
    for z in evals2:
        for x in range(z-last):
            newHomeAway.append(homeAwayPPA[counter2])
            newRepeat.append(repeatPPA[counter2])
            newRobin.append(robinPPA[counter2])
        last = z
        counter2 += 1
            
    homeAwayPPA = newHomeAway
    repeatPPA = newRepeat
    robinPPA = newRobin

    for z in range(len(homeAwayPPA)):
        totalPPA.append(homeAwayPPA[z] + repeatPPA[z] + robinPPA[z])

    plt.rcParams.update({'font.size': 25})

    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.plot(totalHC, color='Green', label='Total')
    ax0.plot(robinHC, color='Red', label='Double Round-Robin')
    ax0.plot(homeAwayHC, color='Blue', label='maxStreak')
    ax0.plot(repeatHC, color='Orange', label='noRepeat')

    ax1.plot(totalPPA, color='Green', label='Total')
    ax1.plot(robinPPA, color='Red', label='Double Round-Robin')
    ax1.plot(homeAwayPPA, color='Blue', label='maxStreak')
    ax1.plot(repeatPPA, color='Orange', label='noRepeat')

    ax1.yaxis.tick_right()

    ax0.set_ylabel('Number of Violations')

    plt.subplots_adjust(wspace=0.02, top=0.917)

    fig.text(0.46, 0.05, 'Evaluations')

    test = fig.legend(['Total', 'Double Round-Robin', 'maxStreak', 'noRepeat'], loc='upper center', ncols=4, edgecolor='black', fontsize="25") 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    ax0.set_ylim(-10, 499)
    ax1.set_ylim(-10, 499)


    ax0.text(0.5, 0.95, 'Hill Climber', fontweight='bold', horizontalalignment='center', transform=ax0.transAxes)
    ax1.text(0.5, 0.95, 'PPA', fontweight='bold', horizontalalignment='center', transform=ax1.transAxes)

    ax0.tick_params(labelsize=16)
    ax1.tick_params(labelsize=16)

    plt.show()

def SAGraphs():
    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SAGG/18/Violations3.txt', 'r')
    SAGG = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySAGG = SAGG[0].astype(int)
    repeatSAGG = SAGG[1].astype(int)
    robinSAGG = SAGG[2].astype(int)

    totalSAGG = []
    for z in range(len(homeAwaySAGG)):
        totalSAGG.append(homeAwaySAGG[z] + repeatSAGG[z] + robinSAGG[z])

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SAGG/18/BestViolations3.txt', 'r')
    SAGGBest = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySAGGBest = SAGGBest[0].astype(int)
    repeatSAGGBest = SAGGBest[1].astype(int)
    robinSAGGBest = SAGGBest[2].astype(int)

    totalSAGGBest = []
    for z in range(len(homeAwaySAGGBest)):
        totalSAGGBest.append(homeAwaySAGGBest[z] + repeatSAGGBest[z] + robinSAGGBest[z])

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SASC/18/Violations3.txt', 'r')
    SASC = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySASC = SASC[0].astype(int)
    repeatSASC = SASC[1].astype(int)
    robinSASC = SASC[2].astype(int)

    totalSASC = []
    for z in range(len(homeAwaySASC)):
        totalSASC.append(homeAwaySASC[z] + repeatSASC[z] + robinSASC[z])

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SASC/18/BestViolations3.txt', 'r')
    SASCBest = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySASCBest = SASCBest[0].astype(int)
    repeatSASCBest = SASCBest[1].astype(int)
    robinSASCBest = SASCBest[2].astype(int)

    totalSASCBest = []
    for z in range(len(homeAwaySASCBest)):
        totalSASCBest.append(homeAwaySASCBest[z] + repeatSASCBest[z] + robinSASCBest[z])

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SALR/18/Violations4.txt', 'r')
    SALR = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySALR = SALR[0].astype(int)
    repeatSALR = SALR[1].astype(int)
    robinSALR = SALR[2].astype(int)

    totalSALR = []
    for z in range(len(homeAwaySALR)):
        totalSALR.append(homeAwaySALR[z] + repeatSALR[z] + robinSALR[z])

    directory = 'E:/NewExperiment/NewResults/'
    file = open(directory + 'SALR/18/BestViolations4.txt', 'r')
    SALRBest = np.loadtxt(file, delimiter=',')
    file.close()

    homeAwaySALRBest = SALRBest[0].astype(int)
    repeatSALRBest = SALRBest[1].astype(int)
    robinSALRBest = SALRBest[2].astype(int)

    totalSALRBest = []
    for z in range(len(homeAwaySALRBest)):
        totalSALRBest.append(homeAwaySALRBest[z] + repeatSALRBest[z] + robinSALRBest[z])

    plt.rcParams.update({'font.size': 20})

    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2)

    ax0.plot(totalSAGG, color='Green', label='Total')
    ax0.plot(robinSAGG, color='Red', label='Double Round-Robin')
    ax0.plot(homeAwaySAGG, color='Blue', label='maxStreak')
    ax0.plot(repeatSAGG, color='Orange', label='noRepeat')

    ax1.plot(totalSAGGBest, color='Green', label='Total')
    ax1.plot(robinSAGGBest, color='Red', label='Double Round-Robin')
    ax1.plot(homeAwaySAGGBest, color='Blue', label='maxStreak')
    ax1.plot(repeatSAGGBest, color='Orange', label='noRepeat')

    ax2.plot(totalSASC, color='Green', label='Total')
    ax2.plot(robinSASC, color='Red', label='Double Round-Robin')
    ax2.plot(homeAwaySASC, color='Blue', label='maxStreak')
    ax2.plot(repeatSASC, color='Orange', label='noRepeat')

    ax3.plot(totalSASCBest, color='Green', label='Total')
    ax3.plot(robinSASCBest, color='Red', label='Double Round-Robin')
    ax3.plot(homeAwaySASCBest, color='Blue', label='maxStreak')
    ax3.plot(repeatSASCBest, color='Orange', label='noRepeat')

    ax4.plot(totalSALR, color='Green', label='Total')
    ax4.plot(robinSALR, color='Red', label='Double Round-Robin')
    ax4.plot(homeAwaySALR, color='Blue', label='maxStreak')
    ax4.plot(repeatSALR, color='Orange', label='noRepeat')

    ax5.plot(totalSALRBest, color='Green', label='Total')
    ax5.plot(robinSALRBest, color='Red', label='Double Round-Robin')
    ax5.plot(homeAwaySALRBest, color='Blue', label='maxStreak')
    ax5.plot(repeatSALRBest, color='Orange', label='noRepeat')

    fig.legend(['Total', 'Double Round-Robin', 'maxStreak', 'noRepeat'], loc='upper center', ncols=4)

    ax1.yaxis.tick_right()
    ax3.yaxis.tick_right()
    ax5.yaxis.tick_right()

    ax0.tick_params(labelsize=20)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)
    ax4.tick_params(labelsize=20)
    ax5.tick_params(labelsize=20)

    ax2.set_ylabel('Number of Violations')

    ax0.text(0.5, 0.89, 'SAGG', fontweight='bold', horizontalalignment='center', transform=ax0.transAxes)
    ax1.text(0.5, 0.89, 'SAGG-Best', fontweight='bold', horizontalalignment='center', transform=ax1.transAxes)
    ax2.text(0.5, 0.89, 'SASC', fontweight='bold', horizontalalignment='center', transform=ax2.transAxes)
    ax3.text(0.5, 0.89, 'SASC-Best', fontweight='bold', horizontalalignment='center', transform=ax3.transAxes)
    ax4.text(0.5, 0.89, 'SALR', fontweight='bold', horizontalalignment='center', transform=ax4.transAxes)
    ax5.text(0.5, 0.89, 'SALR-Best', fontweight='bold', horizontalalignment='center', transform=ax5.transAxes)

    test = fig.legend(['Total', 'Double Round-Robin', 'maxStreak', 'noRepeat'], loc='upper center', ncols=4, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    plt.subplots_adjust(wspace=0.02, top=0.95)

    fig.text(0.47, 0.07, 'Evaluations')

    fig.set_size_inches(19.2, 20)

    fig.savefig('C:/Users/megki_000/Desktop/SA.png')

    plt.show()

def predictEvalsSAGG(ax):
        file = open('E:/NewExperiment/NewResults/SAGG/' + str(42) + '/BestViolations' + str(4) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[0].astype(int)
        repeat = text[1].astype(int)
        robin = text[2].astype(int)

        x = []
        for i in range(1, len(homeAway)+1):
            x.append(i)

        total = []
        for z in range(len(homeAway)):
            total.append(homeAway[z] + repeat[z] + robin[z])

        def func(x, a, b):
            return a+b*np.log(x)
        
        sigma=np.ones(len(total))
        sigma[[200000, -1]] = 0.0001
        
        poptTotal, pcov = curve_fit(func, x, total, p0=[total[0], -1], sigma=sigma)

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        plt.rcParams.update({'font.size': 25})

        #fig, (ax0) = plt.subplots(1, 1)
        
        one = ax.plot(x, total, label='SAGG', color='Purple')
        five = ax.plot(x, func(x, *poptTotal), color='Purple', ls='--', dashes=(5, 5))

        a = str(int(poptTotal[1])) + ' ln(x) + ' + str(int(poptTotal[0]))
        props = dict(boxstyle='round', facecolor='Purple', alpha=0.2)
        plt.text(0.99, 0.2, a, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes, bbox=props, fontsize=14)

        #plt.show()

        intercept = np.exp(-poptTotal[0]/poptTotal[1])
        print(intercept)

def predictEvalsSASC(ax):
        file = open('E:/NewExperiment/NewResults/SASC/' + str(42) + '/BestViolations' + str(0) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[0].astype(int)
        repeat = text[1].astype(int)
        robin = text[2].astype(int)

        x = []
        for i in range(1, len(homeAway)+1):
            x.append(i)

        total = []
        for z in range(len(homeAway)):
            total.append(homeAway[z] + repeat[z] + robin[z])

        def func(x, a, b):
            return a+b*np.log(x)
        
        sigma=np.ones(len(total))
        sigma[[0, -1]] = 0.0001
        
        poptTotal, pcov = curve_fit(func, x, total, p0=[total[0], -1], sigma=sigma)

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        fitTotal = np.poly1d(np.polyfit(x, total, 1))
        #print(fitTotal)
        MSETotal = np.square(np.subtract(total, fitTotal(x))).mean()
        rmseTotal = math.sqrt(MSETotal)
        #print(rmseTotal)        

        plt.rcParams.update({'font.size': 25})

        #fig, (ax0) = plt.subplots(1, 1)
        
        one = ax.plot(x, total, label='SASC', color='Green')
        five = ax.plot(x, func(x, *poptTotal), color='Green', ls='--', dashes=(5, 5))
        #five = ax0.plot(x, fitTotal(x), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))

        a = str(int(poptTotal[1])) + ' ln(x) + ' + str(int(poptTotal[0]))
        props = dict(boxstyle='round', facecolor='Green', alpha=0.2)
        plt.text(0.99, 0.2, a, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes, bbox=props, fontsize=14)

        intercept = np.exp(-poptTotal[0]/poptTotal[1])
        print(intercept)

        #print(fitTotal.coefficients[0])

        intercept = -fitTotal.coefficients[1]/fitTotal.coefficients[0]
        #print(intercept)

        return ax0

def predictEvalsSALR(ax):
        file = open('E:/NewExperiment/NewResults/SALR/' + str(42) + '/BestViolations' + str(1) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[0].astype(int)
        repeat = text[1].astype(int)
        robin = text[2].astype(int)

        x = []
        for i in range(1, len(homeAway)+1):
            x.append(i)

        total = []
        for z in range(len(homeAway)):
            total.append(homeAway[z] + repeat[z] + robin[z])

        def func(x, a, b):
            return a+b*np.log(x)
        
        sigma=np.ones(len(total))
        sigma[[400000, -1]] = 0.0001
        
        poptTotal, pcov = curve_fit(func, x, total, p0=[total[0], -1], sigma=sigma)

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        fitTotal = np.poly1d(np.polyfit(x, total, 1))
        #print(fitTotal)
        MSETotal = np.square(np.subtract(total, fitTotal(x))).mean()
        rmseTotal = math.sqrt(MSETotal)
        #print(rmseTotal)        

        plt.rcParams.update({'font.size': 25})
        
        one = ax.plot(x, total, label='SALR', color='Red')
        five = ax.plot(x, func(x, *poptTotal), color='Red', ls='--', dashes=(5, 5))
        #five = ax0.plot(x, fitTotal(x), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))

        
        a = str(int(poptTotal[1])) + ' ln(x) + ' + str(int(poptTotal[0]))
        props = dict(boxstyle='round', facecolor='Red', alpha=0.2)
        plt.text(0.99, 0.028, a, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes, bbox=props, fontsize=14)

        intercept = np.exp(-poptTotal[0]/poptTotal[1])
        print(intercept)

def predictEvalsHC(ax):
        file = open('E:/NewExperiment/NewResults/HC/' + str(42) + '/Violations' + str(3) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[0].astype(int)
        repeat = text[1].astype(int)
        robin = text[2].astype(int)

        x = []
        for i in range(1, len(homeAway)+1):
            x.append(i)

        total = []
        for z in range(len(homeAway)):
            total.append(homeAway[z] + repeat[z] + robin[z])

        def func(x, a, b):
            return a+b*np.log(x)
        
        sigma=np.ones(len(total))
        #200000, but 400000 for 52 teams!!!!
        sigma[[200000, -1]] = 0.0001
        
        poptTotal, pcov = curve_fit(func, x, total, p0=[total[0], -1], sigma=sigma)

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        fitTotal = np.poly1d(np.polyfit(x, total, 1))
        #print(fitTotal)
        MSETotal = np.square(np.subtract(total, fitTotal(x))).mean()
        rmseTotal = math.sqrt(MSETotal)
        #print(rmseTotal)        

        plt.rcParams.update({'font.size': 25})

        #fig, (ax0) = plt.subplots(1, 1)
        
        one = ax.plot(x, total, label='HC', color='Orange')
        five = ax.plot(x, func(x, *poptTotal), color='Orange', ls='--', dashes=(5, 5))
        #five = ax0.plot(x, fitTotal(x), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))

        a = str(int(poptTotal[1])) + ' ln(x) + ' + str(int(poptTotal[0]))
        props = dict(boxstyle='round', facecolor='Orange', alpha=0.2)
        plt.text(0.99, 0.15, a, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes, bbox=props, fontsize=14)

        intercept = np.exp(-poptTotal[0]/poptTotal[1])
        print(intercept)

def predictEvalsPPA(ax):
        file = open('E:/NewExperiment/NewResults/PPA/' + str(42) + '/Violations' + str(8) + '.txt', 'r')
        text = np.loadtxt(file, delimiter=',')
        file.close()

        homeAway = text[0].astype(int)
        repeat = text[1].astype(int)
        robin = text[2].astype(int)
        evals2 = text[5].astype(int)
        total = []

        newHomeAway = []
        newRepeat = []
        newRobin = []

        counter2 = 0
        last = 0
        for z in evals2:
            for x in range(z-last):
                newHomeAway.append(homeAway[counter2])
                newRepeat.append(repeat[counter2])
                newRobin.append(robin[counter2])
            last = z
            counter2 += 1
            
        homeAway = newHomeAway
        repeat = newRepeat
        robin = newRobin

        for z in range(len(homeAway)):
            total.append(homeAway[z] + repeat[z] + robin[z])

        x = []
        for i in range(1, len(total)+1):
            x.append(i)

        def func(x, a, b):
            return a+b*np.log(x)
        
        sigma=np.ones(len(total))
        #24,30 = 2000000, 36 = 600000, 40+=0
        sigma[[600000, -1]] = 0.0001
        
        poptTotal, pcov = curve_fit(func, x, total, p0=[total[0], -1], sigma=sigma)

        MSETotal  = np.square(np.subtract(total, func(x, *poptTotal))).mean()
        rmseTotal  = math.sqrt(MSETotal)
        #print(rmseTotal)

        fitTotal = np.poly1d(np.polyfit(x, total, 1))
        #print(fitTotal)
        MSETotal = np.square(np.subtract(total, fitTotal(x))).mean()
        rmseTotal = math.sqrt(MSETotal)
        #print(rmseTotal)        

        #fig, (ax0) = plt.subplots(1, 1)

        a = str(int(poptTotal[1])) + ' ln(x) + ' + str(int(poptTotal[0]))
        props = dict(boxstyle='round', facecolor='Blue', alpha=0.2)
        plt.text(0.99, 0.15, a, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes, bbox=props, fontsize=14)
        
        one = ax.plot(x, total, label='PPA', color='Blue')
        five = ax.plot(x, func(x, *poptTotal), color='Blue', ls='--', dashes=(5, 5))
        #five = ax0.plot(x, fitTotal(x), label='Total Fitted', color='Green', ls='--', dashes=(5, 5))

        intercept = np.exp(-poptTotal[0]/poptTotal[1])
        print(intercept)

        return ax

def calculate():
    run = 6
    nrTeams = 18
    nrRounds = 2*(nrTeams-1)
    file = open('E:/NewExperiment/NewResults/HC/' + str(nrTeams) + '/DistanceMatrix.txt', 'r')
    distanceMatrix = np.loadtxt(file, dtype='int')
    file.close()
    
    file = open('E:/NewExperiment/NewResults/HC/' + str(nrTeams) + '/BestSchedule' + str(run) + '.txt', 'r')
    text = np.loadtxt(file, delimiter=',', dtype='int')
    file.close()

    opponent = np.copy(text)

    for i, round in enumerate(opponent):
        newRound = np.copy(round)
        for j, team in enumerate(round):
            if team > 0:
                newRound[j] = j
            else:
                newRound[j] = abs(team)-1
        opponent[i] = newRound

    distance = 0

    for team in range(nrTeams):
        for round in range(nrRounds):
            if round == 0:
                distance += distanceMatrix[team][opponent[round][team]]
            else:
                distance += distanceMatrix[opponent[round-1][team]][opponent[round][team]]
        if opponent[nrRounds-1][team] != team:
            distance += distanceMatrix[opponent[round-1][team]][team]

    print(distance)

def makeRatioGraph():
    HC = [11/11, 11/11, 11/11, 11/11, 11/11, 2/11, 0, 0, 0, 0]
    SASC = [11/11, 11/11, 10/11, 0, 0, 0, 0, 0, 0, 0]
    SALR = [11/11, 11/11, 11/11, 11/11, 0, 0, 0, 0, 0, 0]
    SAGG = [11/11, 11/11, 11/11, 11/11, 9/11, 3/11, 0, 0, 0, 0]
    PPA = [11/11, 6/11, 5/11, 0, 0, 0, 0, 0, 0, 0]

    x = []
    for i in range(6, 61, 6):
        x.append(i)
    
    plt.rcParams.update({'font.size': 25})

    fig, ax1 = plt.subplots(1,1)

    plt.plot(x, HC, label='HC', color='orange')
    plt.plot(x, SASC, label='SASC', color='purple')
    plt.plot(x, SALR, label='SALR', color='green')
    plt.plot(x, SAGG, label='SAGG', color='blue')
    plt.plot(x, PPA, label='PPA', color='red')

    test = fig.legend(['HC', 'SASC', 'SALR', 'SAGG', 'PPA'], loc='upper center', ncols=5, edgecolor='black') 

    test.get_frame().set_alpha(0.5)
    test.get_frame().set_facecolor('wheat')

    plt.xlabel('Number of Teams')
    plt.ylabel('Ratio of Valid Schedules')

    fig.set_size_inches(19.2, 10.8)

    plt.show()

def makeRemainingGraph():
    pass


HCPPAGraph()

#plt.rcParams.update({'font.size': 25})
#fig, (ax0) = plt.subplots(1, 1)

#predictEvalsPPA(ax0)
#predictEvalsHC(ax0)
#predictEvalsSASC(ax0)
#predictEvalsSALR(ax0)
#predictEvalsSAGG(ax0)

#plt.legend(loc='best')
#plt.xlabel('Number of Evaluations ($10^6$)')
#plt.ylabel('Number of Violations')
#plt.show()
