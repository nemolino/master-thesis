from copy import deepcopy
from time import perf_counter
from scipy import stats
import numpy as np
import random
random.seed(42)

class RNG:

    gen = [None] * 71
    for k in range(2,71):
        xk = np.arange(k)
        pk = -xk+k
        pk = pk / np.sum(pk)
        gen[k] = stats.rv_discrete(values=(xk, pk))

    values = [None] * 71
    for k in range(2,71):
        values[k] = gen[k].rvs(size=5000)

class HeuristicPricing:

    def __init__(self, data):

        (
            self.I, self.c, self.P, self.a2,
            self.T, self.D,
            self.Q_nodes_R_core, self.Q_links_R_bandwidth,
            self.q_microservices_R_core, self.q_connections_R_bandwidth,
            self.b_microservices_one, self.b_connections_one
        ) = data
        

        ### SOLUTION 
        
        # capire come rifarlo
        self.SOL_feasible_nodes = [
            sorted(self.b_microservices_one[u], key= lambda i: self.c[i])
            for u in range(self.T)
        ]

        self.SOL_residual_core = deepcopy(self.Q_nodes_R_core)              # I
        self.SOL_residual_bandwidth = deepcopy(self.Q_links_R_bandwidth)    # IxI

        self.SOL_cur_x = np.full((self.T,), -1)     # T
        self.SOL_cur_z = 0

        self.SOL_cur_u_not_fixed = np.full((self.T,), True)     # T
        self.SOL_cur_u_fixed = np.full((self.T,), False)        # T

        # for u in range(self.T):
        #     print(u, self.SOL_feasible_nodes[u])
        # print()
        #print(self.SOL_residual_core.shape, self.SOL_residual_core) 
        #print(self.SOL_residual_bandwidth.shape, self.SOL_residual_bandwidth) 
        #print(self.SOL_cur_x)
        #print(self.SOL_cur_u_not_fixed)
        #print(self.SOL_cur_u_fixed)
        
        ### strutture immutabili

        self.incoming = np.full((self.T,self.T), False)     # TxT
        self.outcoming = np.full((self.T,self.T), False)    # TxT
        for u,v in self.D:
            self.outcoming[u][v] = True
            self.incoming[v][u] = True

        possible_microservices_on_i = np.full((self.I,self.T), False) # IxT
        for u in range(self.T):
            for i in self.SOL_feasible_nodes[u]:
                possible_microservices_on_i[i][u] = True
        self.possible_microservices_on_i = possible_microservices_on_i

        #print(self.incoming)
        #print(self.outcoming)
        #print(self.possible_microservices_on_i)


        # non dovrebbe fallire mai perchè esiste un mapping feasible dell'app
        self.preprocessing() 

        # for u in range(self.T):
        #     print(u, self.SOL_feasible_nodes[u])
        # print()
        # print(self.SOL_residual_core.shape, self.SOL_residual_core) 
        # print(self.SOL_residual_core_RESTART) 
        # print(self.SOL_residual_bandwidth.shape, self.SOL_residual_bandwidth) 
        # print(self.SOL_residual_bandwidth_RESTART) 
        # print(self.SOL_cur_x)
        # print(self.SOL_cur_u_not_fixed)
        # print(self.SOL_cur_u_fixed)

        ### strutture per il restart
        self.SOL_feasible_nodes_RESTART     = deepcopy(self.SOL_feasible_nodes)
        self.SOL_residual_core_RESTART      = np.copy(self.SOL_residual_core)
        self.SOL_residual_bandwidth_RESTART = np.copy(self.SOL_residual_bandwidth)
        self.SOL_cur_x_RESTART              = np.copy(self.SOL_cur_x)
        self.SOL_cur_z_RESTART              = self.SOL_cur_z
        self.SOL_cur_u_not_fixed_RESTART    = np.copy(self.SOL_cur_u_not_fixed)
        self.SOL_cur_u_fixed_RESTART        = np.copy(self.SOL_cur_u_fixed)
        

    def generate_feasible_columns_semigreedy(self):

        start = perf_counter()

        # vediamo se tenerli
        I,c,a2,T,D,q_microservices_R_core,q_connections_R_bandwidth = self.I, self.c, self.a2, self.T, self.D, self.q_microservices_R_core, self.q_connections_R_bandwidth
        P = self.P

        # inizio con la struttura pulita
        self.SOL_feasible_nodes =       deepcopy(self.SOL_feasible_nodes_RESTART)
        self.SOL_residual_core =        np.copy(self.SOL_residual_core_RESTART)             
        self.SOL_residual_bandwidth =   np.copy(self.SOL_residual_bandwidth_RESTART)
        self.SOL_cur_x =                np.copy(self.SOL_cur_x_RESTART)
        self.SOL_cur_z =                self.SOL_cur_z_RESTART
        self.SOL_cur_u_not_fixed =      np.copy(self.SOL_cur_u_not_fixed_RESTART)
        self.SOL_cur_u_fixed =          np.copy(self.SOL_cur_u_fixed_RESTART)

        μ = 0.5     # vedere come va il settaggio
        
        for u in np.nonzero(self.SOL_cur_u_not_fixed)[0]:

            assert len(self.SOL_feasible_nodes[u]) > 0

            # setting up RCL
            min_cost = c[self.SOL_feasible_nodes[u][0]]
            max_cost = c[self.SOL_feasible_nodes[u][-1]]
            t = (1-μ) * min_cost + μ * max_cost
            self.SOL_feasible_nodes[u] = [
                i for i in self.SOL_feasible_nodes[u] if c[i] <= t
            ]
        
        
        err = self.preprocessing()
        if err:
            return None
        
        self.SOL_scanning_order = tuple(sorted(
            #list(range(self.T))
            np.nonzero(self.SOL_cur_u_not_fixed)[0], 
            key=lambda u: len(self.SOL_feasible_nodes[u])
        ))
        
        local_SOL_feasible_nodes_RESTART        = deepcopy(self.SOL_feasible_nodes)
        local_SOL_residual_core_RESTART         = np.copy(self.SOL_residual_core)
        local_SOL_residual_bandwidth_RESTART    = np.copy(self.SOL_residual_bandwidth)
        local_SOL_cur_x_RESTART                 = np.copy(self.SOL_cur_x)
        local_SOL_cur_z_RESTART                 = self.SOL_cur_z
        local_SOL_cur_u_not_fixed_RESTART       = np.copy(self.SOL_cur_u_not_fixed)
        local_SOL_cur_u_fixed_RESTART           = np.copy(self.SOL_cur_u_fixed)
        
        columns = []
        appended = 0
        runs_count = 0
        #k = 1000
        while True:

            if appended >= 30 or perf_counter()-start > 0.05:
                break

            runs_count += 1
            ### restart
            self.SOL_feasible_nodes =       deepcopy(local_SOL_feasible_nodes_RESTART)
            self.SOL_residual_core =        np.copy(local_SOL_residual_core_RESTART)
            self.SOL_residual_bandwidth =   np.copy(local_SOL_residual_bandwidth_RESTART)
            self.SOL_cur_x =                np.copy(local_SOL_cur_x_RESTART)
            self.SOL_cur_z =                local_SOL_cur_z_RESTART
            self.SOL_cur_u_not_fixed =      np.copy(local_SOL_cur_u_not_fixed_RESTART)
            self.SOL_cur_u_fixed =          np.copy(local_SOL_cur_u_fixed_RESTART)
            ###

            err = self.heuristic_greedy_random_try_solve2()
            if err:
                continue

            # building the column
            new_col_q_core = np.zeros(I, dtype=np.int32)
            for u in range(T):
                i = self.SOL_cur_x[u]
                new_col_q_core[i] += q_microservices_R_core[u]

            new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
            for (u,v) in D:
                i = self.SOL_cur_x[u]
                j = self.SOL_cur_x[v]
                for (l,m) in P[i][j]:
                    new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

            assert sum(c[self.SOL_cur_x[u]] for u in range(T)) == self.SOL_cur_z
            columns.append(
                (
                    self.SOL_cur_z,             # col_cost
                    new_col_q_core,             # col_q_core
                    new_col_q_bandwidth,        # col_q_bandwidth 
                    np.copy(self.SOL_cur_x)     # original_x
                )
            ) 
            appended += 1
            
        
        #end = perf_counter()
        #total_time = end-start
        #print(f"runs_count {runs_count} , total time {total_time}")
        return columns if len(columns) > 0 else None
    
    

    def solve_multiple(self, λ_positive, μ_positive, η_n):

        start = perf_counter()

        # vediamo se tenerli
        I,c,a2,T,D,q_microservices_R_core,q_connections_R_bandwidth = self.I, self.c, self.a2, self.T, self.D, self.q_microservices_R_core, self.q_connections_R_bandwidth
        P = self.P

        # inizio con la struttura pulita
        self.SOL_feasible_nodes =       deepcopy(self.SOL_feasible_nodes_RESTART)
        self.SOL_residual_core =        np.copy(self.SOL_residual_core_RESTART)             
        self.SOL_residual_bandwidth =   np.copy(self.SOL_residual_bandwidth_RESTART)
        self.SOL_cur_x =                np.copy(self.SOL_cur_x_RESTART)
        self.SOL_cur_z =                self.SOL_cur_z_RESTART
        self.SOL_cur_u_not_fixed =      np.copy(self.SOL_cur_u_not_fixed_RESTART)
        self.SOL_cur_u_fixed =          np.copy(self.SOL_cur_u_fixed_RESTART)

        c_new = np.copy(c)

        μ = 0.2     # vedere come va il settaggio

        
        for u in np.nonzero(self.SOL_cur_u_not_fixed)[0]:

                # for i in range(I):
                #     c_new[i] = c[i]
                for (i, λ_value) in λ_positive:
                    c_new[i] += q_microservices_R_core[u] * λ_value

                self.SOL_feasible_nodes[u] = sorted(
                    self.SOL_feasible_nodes[u], key= lambda i: c_new[i]
                )
                
                assert len(self.SOL_feasible_nodes[u]) > 0

                # setting up RCL
                min_cost = c_new[self.SOL_feasible_nodes[u][0]]
                max_cost = c_new[self.SOL_feasible_nodes[u][-1]]
                t = (1-μ) * min_cost + μ * max_cost
                self.SOL_feasible_nodes[u] = [
                    i for i in self.SOL_feasible_nodes[u] if c_new[i] <= t
                ]
                
        err = self.preprocessing()
        if err:
            return None
        
        self.SOL_scanning_order = tuple(sorted(
            #list(range(self.T))
            np.nonzero(self.SOL_cur_u_not_fixed)[0], 
            key=lambda u: len(self.SOL_feasible_nodes[u])
        ))
        
        local_SOL_feasible_nodes_RESTART        = deepcopy(self.SOL_feasible_nodes)
        local_SOL_residual_core_RESTART         = np.copy(self.SOL_residual_core)
        local_SOL_residual_bandwidth_RESTART    = np.copy(self.SOL_residual_bandwidth)
        local_SOL_cur_x_RESTART                 = np.copy(self.SOL_cur_x)
        local_SOL_cur_z_RESTART                 = self.SOL_cur_z
        local_SOL_cur_u_not_fixed_RESTART       = np.copy(self.SOL_cur_u_not_fixed)
        local_SOL_cur_u_fixed_RESTART           = np.copy(self.SOL_cur_u_fixed)

        
        columns = []
        appended = 0
        #runs = 0
        while True:
            for _ in range(10):

                #runs += 1
                ### restart
                self.SOL_feasible_nodes =       deepcopy(local_SOL_feasible_nodes_RESTART)
                self.SOL_residual_core =        np.copy(local_SOL_residual_core_RESTART)
                self.SOL_residual_bandwidth =   np.copy(local_SOL_residual_bandwidth_RESTART)
                self.SOL_cur_x =                np.copy(local_SOL_cur_x_RESTART)
                self.SOL_cur_z =                local_SOL_cur_z_RESTART
                self.SOL_cur_u_not_fixed =      np.copy(local_SOL_cur_u_not_fixed_RESTART)
                self.SOL_cur_u_fixed =          np.copy(local_SOL_cur_u_fixed_RESTART)
                ###

                err = self.heuristic_greedy_random_try_solve2()
                if err:
                    continue

                self.SOL_cur_z -= η_n

                if self.SOL_cur_z >= 0:
                    continue
                
                self.SOL_cur_z += sum(
                    q_microservices_R_core[u] * λ_value
                    for (i, λ_value) in λ_positive
                        for u in range(T)
                            if self.SOL_cur_x[u] == i
                )
                
                if self.SOL_cur_z >= 0:
                    continue

                self.SOL_cur_z += sum(
                    q_connections_R_bandwidth[u,v] * μ_value
                    for ((l,m), μ_value) in μ_positive
                        for (i,j) in a2[l][m]
                            for (u,v) in D
                                if self.SOL_cur_x[u] == i and self.SOL_cur_x[v] == j
                )

                if self.SOL_cur_z < -0.001:
                    
                    # building the column
                    new_col_q_core = np.zeros(I, dtype=np.int32)
                    for u in range(T):
                        i = self.SOL_cur_x[u]
                        new_col_q_core[i] += q_microservices_R_core[u]

                    new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
                    for (u,v) in D:
                        i = self.SOL_cur_x[u]
                        j = self.SOL_cur_x[v]
                        for (l,m) in P[i][j]:
                            new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

                    columns.append(
                        (
                            sum(c[self.SOL_cur_x[u]] for u in range(T)),    # col_cost
                            new_col_q_core,                                 # col_q_core
                            new_col_q_bandwidth,                            # col_q_bandwidth 
                            np.copy(self.SOL_cur_x)                         # original_x
                        )
                    ) 
                    appended += 1
            
            if appended >= 10 or perf_counter()-start > 0.1:
                break
        
        #print(runs)
        return columns if len(columns) > 0 else None
    

    def preprocessing(self):

        while True:
            found = False

            for u in np.nonzero(self.SOL_cur_u_not_fixed)[0]:

                if len(self.SOL_feasible_nodes[u]) == 1:
                    found = True

                    # assign_microservice_u_to_unique_candidate
                    i = self.SOL_feasible_nodes[u][0]

                    err = self.assign_microservice_u_to_node_i(u,i)
                    if err: 
                        return True
                    
            if not found: 
                break
        
        return False
    

    def assign_microservice_u_to_node_i(self, u, i):
        
        # questi possono essere spostati più giù o eliminati, sono per comodità
        c = self.c
        T = self.T
        q_microservices_R_core = self.q_microservices_R_core
        b_connections_one = self.b_connections_one
        P = self.P
        q_connections_R_bandwidth = self.q_connections_R_bandwidth

        ###### FISSAGGIO

        self.SOL_feasible_nodes[u] = []
        self.SOL_cur_x[u] = i
        self.SOL_cur_z += c[i]
        self.SOL_cur_u_not_fixed[u] = False
        self.SOL_cur_u_fixed[u] = True

        ###### EFFETTI COLLATERALI

        ### consumo core

        self.SOL_residual_core[i] -= q_microservices_R_core[u]
        #self.SOL_residual_core_RESTART[i] += q_microservices_R_core[u]
        if self.SOL_residual_core[i] < 0:
            #raise RuntimeError(f"FAIL ... node {i} has not enough cores to hold microservice {u}")
            return True
        
        for v in np.nonzero(np.logical_and(
            self.SOL_cur_u_not_fixed, self.possible_microservices_on_i[i]))[0]:

            if self.SOL_residual_core[i] < q_microservices_R_core[v]: 
                if i in self.SOL_feasible_nodes[v]:
                    self.SOL_feasible_nodes[v].remove(i)
                if len(self.SOL_feasible_nodes[v]) == 0:
                    #raise RuntimeError(f"FAIL ... elimination of node {i} from {v} candidates leaves microservice {v} with 0 candidates")
                    return True

        ###

        for v in np.nonzero(np.logical_and(
            self.outcoming[u], self.SOL_cur_u_not_fixed))[0]:
            
            self.SOL_feasible_nodes[v] = [
                k for k in self.SOL_feasible_nodes[v] if (i,k) in b_connections_one[u][v]
            ]
            if len(self.SOL_feasible_nodes[v]) == 0:
                #raise RuntimeError(f"FAIL ... elimination using b_(u,_)^(i,_) on arc (u,v) leaves microservice {v} with 0 candidates")
                return True
        
        for v in np.nonzero(np.logical_and(
            self.incoming[u], self.SOL_cur_u_not_fixed))[0]:

            self.SOL_feasible_nodes[v] = [
                k for k in self.SOL_feasible_nodes[v] if (k,i) in b_connections_one[v][u]
            ]
            if len(self.SOL_feasible_nodes[v]) == 0:
                #raise RuntimeError(f"FAIL ... elimination using b_(_,u)^(_,i) on arc (v,u) leaves microservice {v} with 0 candidates")
                return True

        ### consumo bandwidth
        
        for v in np.nonzero(np.logical_and(
            self.outcoming[u], self.SOL_cur_u_fixed))[0]:

            assert self.SOL_cur_x[v] != -1

            for i,j in P[i][self.SOL_cur_x[v]]:
                
                self.SOL_residual_bandwidth[i,j] -= q_connections_R_bandwidth[u,v]
                #self.SOL_residual_bandwidth_RESTART[i,j] += q_connections_R_bandwidth[u,v]
                if self.SOL_residual_bandwidth[i,j] < 0:
                    #raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({u},{v})")
                    return True
                
        for v in np.nonzero(np.logical_and(
            self.incoming[u], self.SOL_cur_u_fixed))[0]:

            assert self.SOL_cur_x[v] != -1

            for i,j in P[self.SOL_cur_x[v]][i]:

                self.SOL_residual_bandwidth[i,j] -= q_connections_R_bandwidth[v,u]
                #self.SOL_residual_bandwidth_RESTART[i,j] += q_connections_R_bandwidth[v,u]
                if self.SOL_residual_bandwidth[i,j] < 0:
                    #raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({v},{u})")
                    return True
        
        return False
    

    def heuristic_greedy_random_try_solve2(self):

        for u in self.SOL_scanning_order:

            if self.SOL_cur_u_not_fixed[u]:

                l = len(self.SOL_feasible_nodes[u])
                
                assert l > 1
                
                #i = random.choice(sol.feasible_nodes[u])

                idx = random.choice(RNG.values[l]) #RNG.get(l)  
                i = self.SOL_feasible_nodes[u][idx]
                
                err = self.assign_microservice_u_to_node_i(u,i)
                if err:
                    return True
        
                err = self.preprocessing()
                if err:
                    return True

        return False