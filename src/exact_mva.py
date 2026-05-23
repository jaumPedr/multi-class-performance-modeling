#D --> (Di,r) average service demand of class r customers at device i
#N --> (Nr) population of class r customers

#K --> number of devices or service centers
#R --> number of customer classes

#Z --> (Zr) think time of class r customers
#X --> (Xr) throughput of class r customers

#n_Device --> (ni) average number of customers at each device
#N_state --> (_N) population vector (N1, N2, ..., Nr)

#previous_N --> (_N - 1r) previous population state
#previous_n --> n_Device values for state (_N - 1r)

#R_residence --> (Ri,r) average residence time of class r customers at device i

#states --> stores previously computed Exact MVA states
#current_n_Device --> n_Device values for current state _N


import json
import numpy as np

def exact_MVA_Start(D: list, N: list, Z: list | 0 = 0):

    #Convert inputs to numpy arrays
    D = np.array(D, dtype=float)
    N = np.array(N, dtype=int)

    K, R = D.shape

    #Default think time
    if isinstance(Z, int) and Z == 0:
        Z = np.zeros(R)

    Z = np.array(Z, dtype=float)

    #Stores previously computed states
    states = {}
    states[tuple(np.zeros(R, dtype=int))] = {
        "X": np.zeros(R),
        "n_Device": np.zeros(K)
    }


    j = []
    results = []
    results = exact_MVA_InterationLoop(D, N, K, R, Z, 0, j, states, results)
    return results

def exact_MVA_InterationLoop( D: list, N: list, K: int, R: int, Z: list, index_N: int, j: list, states : dict, results : list):

    #Complete population state generated
    if index_N == len(N):

        N_state = np.array(j)

        if not np.all(N_state == 0):
            results.append(exact_MVA_Algorithm(D, R, K, Z, N_state, states))

        return results

    #Generate all possible states recursively
    for i in range(N[index_N] + 1):
        j.append(i)
        exact_MVA_InterationLoop(D, N, K, R, Z, index_N + 1, j, states, results)

        j.pop()

    return results


def exact_MVA_Algorithm(D: list, R: int, K: int, Z, N_state: list, states: dict):

    R_residence = np.zeros((K, R))
    X = np.zeros(R)

    for r in range(R):

        previous_N = N_state.copy()

        previous_N[r] -= 1

        previous_key = tuple(previous_N)

        if N_state[r] > 0:
            
            #Recover previous queue lengths
            if previous_key in states:
                previous_n = states[previous_key]["n_Device"]

            else:
                previous_n = np.zeros(K)

            #Residence time calculation
            for k in range(K):
                R_residence[k][r] = ( D[k, r] * (1 + previous_n[k]) )

        else:
            for k in range(K):
                R_residence[k][r] = 0

        #Throughput calculation
        if ( Z[r] + np.sum(R_residence[:, r])) == 0:
            X[r] = 0

        else:  
            X[r] = ( N_state[r] / ( Z[r] + np.sum(R_residence[:, r])) )

    #Average customers per device
    current_n_Device = np.zeros(K)

    for k in range(K):
        current_n_Device[k] = np.sum( X * R_residence[k] )

    #Save current state
    states[tuple(N_state)] = {
        "X": X.copy(),
        "n_Device": current_n_Device.copy()
    }

    
    result = {
        "N_State": N_state.tolist(),
        "System_Throughput_Per_Classes": X.tolist(),
        "Average_Number_Of_Customers_At_Devices": current_n_Device.tolist(),
        "Residence_Time": R_residence.tolist()
    }
    return result


#Test Values
D = [
    [0.375, 0.105],
    [0.480, 0.180],
    [0.240, 0.0]
    ]

N = [1, 3]

res = exact_MVA_Start(D, N)

print(json.dumps(res, indent=4))
