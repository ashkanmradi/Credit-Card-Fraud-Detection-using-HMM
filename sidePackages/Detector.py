import numpy as np
from sidePackages.HMM import HMM


class Detector:
    threshold = 0
    def __init__(self,HMMModel):
        self.hmm = HMMModel
    def setThreshold(self, threshold):
        self.threshold = threshold

    def getThreshold(self):
        return self.threshold

    def detect(self, o):
        fwd = self.hmm.forwardProc(o)
        bwd = self.hmm.backwardProc(o)
        result = np.zeros(len(o))
        #print(len(o))

        for i in range(len(o)):
            result[i] = 0
            for j in range(self.hmm.numStates):
                result[i] = result[i] + fwd[j, i] * bwd[j, i]

        return result

    def fraudEvaluation(self, alpha, newTransaction):
        risk = False
        if(alpha > self.threshold):
            risk = True

        return risk

    def calculateAlpha(self, oldTransaction, newTransaction):
        alpha = 0
        result = self.detect(oldTransaction)
        newResult = self.detect(newTransaction)

        # print("newResult=", newResult[0], " ---- old result=", result[0])

        difference = result[0] - newResult[0]
        alpha = difference / result[0]
        print("delta_alpha=", alpha)
        return alpha


    def calculateOrdinaryAlpha(self, oldTransaction, newTransaction):
        alpha = 0
        fwdOld = self.hmm.forwardProc(oldTransaction)
        fwdNew = self.hmm.forwardProc(newTransaction)
        alpha1 = fwdOld[-1].sum()
        alpha2 = fwdNew[-1].sum()

        # print("newResult=", alpha2, " ---- oldResult=", alpha1)

        difference = alpha1 - alpha2
        alpha = difference / alpha1
        print("delta_alpha=", alpha)
        return alpha
