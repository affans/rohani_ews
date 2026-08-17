import numpy as np




def MovingWindowAverage(data, window):
    '''
    This returns the sum of the datapoints. Does not take into account the stepsize.
    :param data:
    :param window:
    :return:
    '''

    #dataavg = np.array([np.sum(data[i-window+1:i+1])/window for i in range(0,len(data))])
    #for i in range(window-1):
    #    dataavg[i] = np.nan

    def rolling_window(a):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    dataavg = np.concatenate((np.array([np.nan]*(window-1)),
                              np.mean(rolling_window(data), -1)))

    return dataavg


def MovingAC(data,window,lag):
	#needs fixing
	y = np.concatenate([np.zeros(lag), data[:-lag]])
	cfunc = MovingWindowAverage(data*y,window)
	mu1 = MovingWindowAverage(data,window)
	#mu2 = MovingWindowAverage(data**2, window)
	var = MovingWindowAverage((data-mu1)**2, window)

	mu_y = np.concatenate([np.zeros(lag), mu1[:-lag]])
	var_y = np.concatenate([np.zeros(lag), var[:-lag]])
	AC = (cfunc - mu1*mu_y)/np.sqrt(var*var_y)
	return(AC)

