import numpy as np

class HMM:
    def __init__(self, numStates = 6, sigmaSize = 3):
        self.numStates = numStates
        self.sigmaSize = sigmaSize
        self.a = np.zeros([self.numStates, self.numStates])
        self.b = np.zeros([self.numStates, self.sigmaSize])
        self.pi = np.zeros(self.numStates)

    def initializeHMM(self):
        self.pi = np.random.dirichlet(np.ones(self.numStates))
        self.a = np.random.dirichlet(np.ones(self.numStates), size=self.numStates)
        self.b = np.random.dirichlet(np.ones(self.sigmaSize), size=self.numStates)
        #
        # for i in range(self.numStates):
        #     self.pi[i] = 1.0/ self.numStates
        #
        # for i in range(self.numStates):
        #     for j in range(self.numStates):
        #         self.a[i,j] = 1.0 / self.numStates
        #
        # for i in range(self.numStates):
        #     for j in range(self.sigmaSize):
        #         self.b[i,j] = 1.0 / self.sigmaSize
        #

    def train(self, o, steps):
        T = len(o)
        a1 = np.zeros([self.numStates, self.numStates])
        b1 = np.zeros([self.numStates, self.sigmaSize])
        pi1 = np.zeros(self.numStates)

        for s in range(steps):
            # calculation of Forward and Backward Variables from the current model
            fwd = self.forwardProc(o)
            bwd = self.backwardProc(o)

            # re-estimation of initial state probabilities
            for i in range(self.numStates):
                pi1[i] = self.gamma(i, 0, o, fwd, bwd)

            # re-estimation of transition probabilities
            for i in range(self.numStates):
                for j in range(self.numStates):
                    num = 0
                    denominator = 0
                    for t in range(T):
                        num += self.p(t, i, j, o, fwd, bwd)
                        denominator += self.gamma(i, t, o, fwd, bwd)

                    a1[i,j] = self.divide(num, denominator)

            # re-estimation of emission probabilities
            for i in range(self.numStates):
                for k in range(self.sigmaSize):
                    num = 0
                    denominator = 0
                    for t in range(T):
                        g = self.gamma(i, t, o, fwd, bwd)
                        if(k == o[t]):
                            num += g * 1
                        else:
                            num += g * 0
                        denominator += g

                    b1[i, k] = self.divide(num, denominator)

            self.pi = pi1
            self.a = a1
            self.b = b1


    def forwardProc(self, o):
        T = len(o)
        forward = np.zeros([self.numStates, T])

        # initialization (time 0)
        for i in range(self.numStates):
            forward[i, 0] = self.pi[i] * self.b[i, int(o[0])]

        # induction
        for t in range(T-1):
            for j in range(self.numStates):
                forward[j, t+1] = 0
                for i in range(self.numStates):
                    forward[j, t+1] += (forward[i, t] * self.a[i, j])

                forward[j, t+1] *= self.b[j, int(o[t+1])]

        return forward

    def backwardProc(self, o):
        T = len(o)
        backward = np.zeros([self.numStates, T])

        # initialization (time 0)
        for i in range(self.numStates):
            backward[i, T-1] = 1

        # induction
        for t in range(T-2, -1, -1):
            for i in range(self.numStates):
                backward[i, t] = 0
                for j in range(self.numStates):
                    backward[i, t] += (backward[j, t+1] * self.a[i, j] * self.b[j, int(o[t+1])])

        return backward

    # calculation of probability P(X_t = s_i, X_t+1 = s_j | O, m). Epsilon
    def p(self, t, i, j, o, fwd, bwd):
        if(t == len(o)-1):
            num = fwd[i, t] * self.a[i, j]
        else:
            num = fwd[i, t] * self.a[i, j] * self.b[j, int(o[t+1])] * bwd[j, t+1]

        denominator = 0
        for k in range(self.numStates):
            denominator += (fwd[k, t] * bwd[k, t])

        return self.divide(num, denominator)

    # computes gamma(i, t)
    def gamma(self, i, t, o, fwd, bwd):
        num = fwd[i, t] * bwd[i, t]
        denominator = 0

        for j in range(self.numStates):
            denominator += fwd[j, t] * bwd[j, t]

        return self.divide(num, denominator)


    def divide(self, n, d):
        if(n == 0):
            return 0
        else:
            return (n/d)

