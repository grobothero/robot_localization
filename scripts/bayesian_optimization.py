#!/usr/bin/env python

from robot_localization.srv import para_tuning, para_tuningResponse
import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm
import rospy
import os

filepath = os.path.dirname(os.path.realpath(__file__))+"/"

class bayesian_optimization:
    def __init__(self):
        # Set a counter 
        self.counter = 0
        self.paras_org = np.zeros(3)
        self.paras_next = np.zeros(3)
        self.bo_srv = rospy.Service('parameters_tuning', para_tuning, self.optimization_callback)
        #Initialize optimization kernel
        self.kernel_ = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level = 0.05**2)
        self.gpr = GaussianProcessRegressor(kernel = self.kernel_, n_restarts_optimizer=5)
        self.bounds = np.array([[math.log(1e-7), math.log(1e-2)]])
        self.X_init = []
        self.Y_init = []

    def optimization_callback(self,req):
        print("Optimization starts!")
        print("The pose error is: " + str(req.avr_error))
        print("Current parameters are: " + str(req.current_paras))
        #Get inital samples by line search
        if (self.counter == 0):
            self.paras_org = np.array(req.current_paras)
            self.paras_next = self.paras_org
            self.counter += 1
        elif (self.counter > 0 and self.counter <= 3):
            self.paras_next = self.paras_org / math.pow(10.0, self.counter)
            self.paras_next[2] = self.paras_next[0] * 1.5
            self.counter += 1
        elif (self.counter > 3 and self.counter <= 5):
            self.paras_next = self.paras_org * math.pow(10.0, (self.counter - 3))
            self.paras_next[2] = self.paras_next[0] * 1.5
            self.counter += 1
        
        if (self.counter > 0 and self.counter <= 6):
            self.X_init.append(np.array(req.current_paras))
            self.Y_init.append(math.sqrt(req.avr_error))

        if (self.counter >= 6):
            if (self.counter == 6):
                #Stack all the initial samples
                #X_samples represents the different sets of parameters
                #Y_samples represents the corresponding errors
                self.X_sample = np.vstack(self.X_init)
                self.Y_sample = np.hstack(self.Y_init)
                #Suppose now we just tune one parameter of position in x axis
                #It can be easily modified to tune multiple parameters
                self.X_sample_ = self.X_sample[:, 0].reshape(-1, 1)
                #Take the log so the x axis in proper scale
                self.X_sample_ = np.log(self.X_sample_ )
                self.counter += 1
            else:
                #Continue adding new samples to current samples
                X_next = np.array(req.current_paras)
                Y_next = math.sqrt(req.avr_error)
                self.X_sample = np.vstack((self.X_sample, X_next))
                self.Y_sample = np.append(self.Y_sample, Y_next)
                self.X_sample_ = self.X_sample[:, 0].reshape(-1, 1)
                self.X_sample_ = np.log(self.X_sample_ )
                self.counter += 1

            #Using samples to fit a surrogate model which represents the EKF pose error curve
            self.gpr.fit(self.X_sample_, self.Y_sample)
            #l = self.gpr.kernel_.k1.k2.get_params()['length_scale']
            #sigma_f = np.sqrt(self.gpr.kernel_.k1.k1.get_params()['constant_value'])
            #sigma_y = np.sqrt(self.gpr.kernel_.k2.get_params()['noise_level'])

            #Find the next set of parameters by maximizing the expected improvement
            #The expected improvement is a function to balance the minimum of the fitted surrogate model and uncertainty areas
            X_new = self.propose_location(self.expected_improvement, self.X_sample_, self.Y_sample, self.gpr, self.bounds)

            #Currently we set the constraint as x = y = theta / 1.5
            X_new = np.exp(X_new.ravel())
            self.paras_next[0] = X_new[0]
            self.paras_next[1] = X_new[0]
            self.paras_next[2] = self.paras_next[1] * 1.5
        
        if (self.counter % 10 == 0):
            #Save the sets of parameters during the optimiztion process
            print("The data is saved under the path: " + filepath)
            np.save(filepath + "X_Sample", self.X_sample)
            np.save(filepath + "Y_Sample", self.Y_sample)

        return para_tuningResponse([self.paras_next[0], self.paras_next[1], self.paras_next[2]])

    def expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.1):
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.min(mu_sample)

        with np.errstate(divide='warn'):
            #Large xi leads to more exploration around the uncertainty area
            #Small xi leads to more exploitation around the minimum of the model
            imp = mu_sample_opt - mu.reshape(-1, 1) - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def propose_location(self, acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
        dim = X_sample.shape[1]
        min_val = 1
        min_x = None
        
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')   
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           
                
        return min_x.reshape(1, -1)

if __name__ == "__main__":
    rospy.init_node('bayesian_optimization')
    print("Optimization node is running!")
    bo_ = bayesian_optimization()
    rospy.spin()

