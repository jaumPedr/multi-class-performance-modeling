#D --> (Di,r) average service demand of class r customers at device i
#N --> (Nr) population of closed class r customers

#lambda_r --> arrival rate of open class r customers

#K --> number of devices or service centers
#R --> number of customer classes

#U_open --> utilization caused by open classes
#n_open --> average number of open customers at devices
#n_closed --> average number of closed customers at devices

#R_residence --> (Ri,r) residence time of class r customers at device i

#O --> dictionary storing open classes information
#C --> dictionary storing closed classes information

import json
import numpy as np
import exact_mva as exact

def mixed_mva(D: list, N: list, lambda_r: list):

    #Convert inputs to numpy arrays
    D = np.array(D, dtype=float)
    N = np.array(N, dtype=int)
    lambda_r = np.array(lambda_r, dtype=float)
    
    K, R = D.shape

    O = {}
    C = {}

    #Separate open and closed classes
    for r in range(R):
        if N[r] > 0:
            C[r] = {
                'D_e': np.zeros(shape=K)
            }
            continue
        if lambda_r[r] > 0:
            O[r] = {
                'U': np.zeros(shape=K)
            }

    #Open classes utilization
    U_open = np.zeros(K)
    for r  in O.keys():
        O[r]['U'] = lambda_r[r]* D[:,r]
        U_open += O[r]['U']
    

    #Equivalent demand for closed classes
    for r in C.keys():
        C[r]['D_e'] = D[:,r] / (1 - U_open)


    #Run Exact MVA for closed classes
    D_imput = np.array([C[r]['D_e'] for r in C.keys()]).T
    N_imput = [N[r] for r in C.keys()]
    mva_Closed_Model_Results = exact.exact_MVA_Start(D_imput, N_imput)[-1]
    
    X_Closed_Model = mva_Closed_Model_Results['System_Throughput_Per_Classes']
    current_n_Device_Closed_Model = mva_Closed_Model_Results['Average_Number_Of_Customers_At_Devices']
    
    n_closed = np.zeros(K)

    #Closed customers at devices
    n_closed += current_n_Device_Closed_Model

    R_residence = np.zeros((K, R))

    #Mixed residence times
    for k in range(K):
        for r in range(R):
            if (1 - U_open[k]) == 0: 
                R_residence[k, r] = 0

            else:
                R_residence[k, r] = ( D[k, r] * (1 + n_closed[k]) ) / (1 - U_open[k])

    #Open customers at devices
    n_open = np.zeros(K)

    for k in range(K):
        if U_open[k] < 1:
            n_open[k] = U_open[k] / (1 - U_open[k])
        else:
            n_open[k] = np.inf

        #Total customers at devices
    n_total = n_closed + n_open


    #results
    result = {

        "Open_Classes": {
            int(r): {
                "Utilization": O[r]['U'].tolist()
            }
            for r in O.keys()
        },
        "Closed_Classes": {
            int(r): {
                "Equivalent_Demand": C[r]['D_e'].tolist()
            }
            for r in C.keys()
        },
        "Open_Utilization_Per_Device": U_open.tolist(),
        "Residence_Time": R_residence.tolist(),
        "Closed_Model_Throughput": X_Closed_Model.tolist(),
        "Closed_Customers_Per_Device": n_closed.tolist(),
        "Open_Customers_Per_Device": n_open.tolist(),
        "Total_Customers_Per_Device": n_total.tolist()
    }

    return result