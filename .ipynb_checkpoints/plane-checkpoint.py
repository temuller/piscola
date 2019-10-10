from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
    
def fit_plane(X, Xerr, p0=None, params_names=None, plot=False, print_res=False):
    """ Fits a plane in N dimension given by the shape of the array.
    xxx
    
    """
    
    def log_prob_D(params):
        N = len(params)
        m = params[:N-1]
        b = params[N-1]
        #m = params[:N-2]
        #b, log_lambda = params[N-2:]
        v = np.append(-m, 1.0)

        Sigma2 = np.dot(np.dot(S, v), v)
        #Sigma2 = np.dot(np.dot(S, v), v) + np.exp(2*log_lambda)
        Delta = np.dot(X, v) - b
        
        # Compute the log likelihood up to a constant.
        ll = -0.5 * np.sum(Delta**2 / Sigma2 + np.log(Sigma2))
        return ll
    
    N = X.shape[0]  # number of data points
    d = X.shape[1]  # dimension [e.g., d=2 -> line]
          
    S = np.zeros((N, d, d))
    Xwa =  np.average(X, axis=0,weights=1/Xerr**2)  # weighted average of the variables
    for n in range(N):
        L = np.zeros((d, d))
        err = Xerr[n]
        ###
        w = 1/err**2
        XXw = (X[n]-Xwa)*w
        #S[n] = XXw.T.dot(XXw)/w.T.dot(w)
        S[n] = np.outer(XXw, XXw)/np.outer(w, w)
        ###
        #L[np.diag_indices_from(L)] = np.asarray([err[i] for i in range(d)])
        #S[n] = np.dot(L, L.T)
      
    if params_names is None:
            params_names = [r'm$_%i$'%(i+1) for i in range(d-1)] + ['b']
            #params_names = [r'x$_%i$'%i for i in range(d)]
        
    if plot:
        
        fig, axes = plt.subplots(d-1, d-1, figsize=(2.5*d, 2.5*d))
        for xi, yi in product(range(d), range(d)):
            if yi <= xi:
                continue
            if d!=2:
                ax = axes[yi-1, xi]
            else:
                ax = axes
            ax.errorbar(X.T[xi], X.T[yi], xerr=Xerr.T[xi], yerr=Xerr.T[yi], fmt='.')

        if d==2:
            ax.set_xlabel(params_names[0])
            ax.set_ylabel(params_names[1])
            plt.show()
        
        elif d==3:
            # Make the plots look nicer...
            ax = axes[0, 1]
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[0, 0]
            ax.set_ylabel(params_names[1])
            ax.set_xticklabels([])
            ax = axes[1, 0]
            ax.set_xlabel(params_names[0])
            ax.set_ylabel(params_names[2])
            ax = axes[1, 1]
            ax.set_xlabel(params_names[1])
            ax.set_yticklabels([])
            fig.subplots_adjust(wspace=0, hspace=0)
            plt.show()
            
    # Run the MCMC.
    if p0 is None:
        p0 = np.ones(d)
        #p0 = np.ones(d+1)
    init_p0 = p0.copy()
    
    nwalkers = 100
    sampler_D = emcee.EnsembleSampler(nwalkers, d, log_prob_D)
    #sampler_D = emcee.EnsembleSampler(nwalkers, d+1, log_prob_D)
    p0 = p0 + 1e-4 * np.random.randn(nwalkers, len(p0))
    pos, _, _ = sampler_D.run_mcmc(p0, 500)
    sampler_D.reset()
    sampler_D.run_mcmc(pos, 3000)
    samples_D = sampler_D.flatchain

    params_names = [r'm$_%i$'%(i+1) for i in range(d-1)] + ['b']
    #params_names = [r'm$_%i$'%(i+1) for i in range(d)] + ['b']
    
    if plot:
        corner.corner(samples_D, labels=params_names, truths=init_p0);
        
    res_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_D, [16, 50, 84],
                                                axis=0))))

    if print_res:
        
        fig, ax = plt.subplots(figsize=(1, .1))
        ax.set_frame_on(False)
        for i, param in enumerate(res_mcmc):
            ax.text(4*i, 1, params_names[i] + r' $= %.2f^{+%.2f}_{-%.2f}$'%(param))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        
    return res_mcmc