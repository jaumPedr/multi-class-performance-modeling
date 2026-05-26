#D --> (Di,r) average service demand of class r customers at device i
#N --> (Nr) population of class r customers

#K --> number of devices or service centers
#R --> number of customer classes

#Z --> (Zr) think time of class r customers
#X --> (Xr) throughput of class r customers

#N_state --> (_N) population vector (N1, N2, ..., Nr)

#previous_N --> (_N - 1r) previous population state

#R_residence --> (Ri,r) average residence time of class r customers at device i

#n_Approx --> approximate average number of customers per device/class
#n_Device --> previous iteration queue lengths

import json
import numpy as np

def approximate_MVA_Algorithm(D: list, N: list, err: float, Z: list | int = 0):
    
    #Convert inputs to numpy arrays
    D = np.array(D, dtype=float)
    N = np.array(N, dtype=int)
    
    K, R = D.shape

        #Default think time
    if isinstance(Z, int) and Z == 0:
        Z = np.zeros(R)

    N_state = np.array(N, dtype=int)
    N_state_tuple = tuple(N_state)

    #Residence time and throughput vectors
    R_residence = np.zeros(shape=(K, R), dtype = float)
    X = np.zeros(shape=R, dtype = float)

    #Stores iteration results
    result = []

    #Stores current approximate queue lengths
    state = {}
    state[N_state_tuple]={
        "n_Approx" : np.zeros(shape=(K,R), dtype=float),
        "n_Device": np.zeros(shape=(K,R), dtype=float),
    }

    #Initial approximation
    for r in range(R):
        for  k in range(K):
            if D[k][r] > 0:
                state[N_state_tuple]['n_Approx'][k][r] = N[r]/K
    
    #AMVA iteration loop
    while True:
        
        #Store previous iteration values
        state[N_state_tuple]['n_Device'] = state[N_state_tuple]['n_Approx'].copy()

        #Build approximated state (_N - 1r)
        for r in range(R):
            previous_N = N_state.copy()
            previous_N[r] -= 1
            previous_N_tuple = tuple(previous_N)

            state[previous_N_tuple] = {
            "n_Device": state[N_state_tuple]['n_Device'].copy()
            }

            #Adjust queue length for class r
            for k in range(K):    
                state[previous_N_tuple]['n_Device'][k][r] = ( ((N[r] - 1) * state[N_state_tuple]['n_Device'][k][r]) / N[r] )
        
        #Residence time and throughput calculation
        for r in range(R):
            previous_N = N_state.copy()
            previous_N[r] -= 1
            previous_N_tuple = tuple(previous_N)

            for k in range(K):
                R_residence[k][r] = D[k][r] * ( 1 + sum(state[previous_N_tuple]['n_Device'][k]) )
            
            X[r] = N[r] / (Z[r] + sum(R_residence[:, r]))

        #Update approximate queue lengths
        for r in range(R):
            for k in range(K):
                state[N_state_tuple]['n_Approx'][k][r] = X[r] * R_residence[k][r]
        
        result.append({
            "Iteration": len(result) + 1,
            "N_State": N_state.tolist(),
            "System_Throughput_Per_Classes": X.tolist(),
            "Average_Number_Of_Customers_At_Devices": state[N_state_tuple]['n_Approx'].tolist(),
            "Residence_Time": R_residence.tolist()
        })

        #Denominator adjusted to prevent division by zero
        den = np.maximum(state[N_state_tuple]['n_Approx'], 1e-10)
        
        #Convergence test
        if np.max( np.abs( (state[N_state_tuple]['n_Approx'] - state[N_state_tuple]['n_Device']) / den)) < err:
            break


    return result
    
#Test Values

D = [
    [0.0030, 0.0035],
    [0.0009, 0.0215],
    [0.0123, 0.0011]
    ]

N = [20, 25]

res = approximate_MVA_Algorithm(D, N, 0.01)

with open("./src/scripts_results/approximate_mva.txt", "w") as f:
    f.write("\n".join(map(str, res)))

