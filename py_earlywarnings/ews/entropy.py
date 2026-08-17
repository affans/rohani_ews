import numpy as np



## Returns the entropy of the probability distribution as a list.
# Note this currently only works for integer data.
def MovingEntropy(x, w):

    ## Returns a 2-d numpy array prob with element prob[i,j] returning the
    ## probability of being in state j + min(x) at time t[i]. Assumes timesteps are equal.
    def MovingProb(x, w):

        #def myround(x):
        #    return np.round(base * np.round(x / base), prec)
        xr = np.round(x)
        prob = np.zeros((len(x), int(np.nanmax(xr) - np.nanmin(xr)) + 1))
        y = xr - np.nanmin(xr)
        yint = y.astype(int)
        for i in range(0, w):
            if not np.isnan(y[i]):
                prob[w-1, yint[i]] = prob[w-1, yint[i]] + 1. / w
        for i in range(w, len(x)):
            prob[i, :] = prob[i - 1, :]
            if not np.isnan(y[i]):
                prob[i, yint[i]] = prob[i, yint[i]] + 1. / w
            if not np.isnan(y[i-w]):
                prob[i, yint[i-w]] = prob[i, yint[i-w]] - 1. / w
        return prob

    prob = MovingProb(x, w)
    ent = [np.nan]*(w-1)
    for i in range(w-1, prob.shape[0]):
        ent.append( -np.nansum(prob[i,]*np.log(prob[i,])))
    return np.array(ent)

