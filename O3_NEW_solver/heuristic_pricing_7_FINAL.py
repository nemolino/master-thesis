from copy import deepcopy
import numpy as np
from time import perf_counter
from scipy import stats
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


class Solution():

    def __init__(self, data):

        (
            I, c, _, _,
            T, D,
            Q_nodes_R_core, Q_links_R_bandwidth,
            _, _,
            b_microservices_one, _
        ) = data

        self.feasible_nodes = [
            sorted(b_microservices_one[u], key= lambda i: c[i])
            for u in range(T)
        ]
        
        self.residual_core = deepcopy(Q_nodes_R_core)
        self.residual_bandwidth = deepcopy(Q_links_R_bandwidth)
        #self.d_still_to_check = D.copy()
        self.cur_x = [None] * T
        self.cur_z = 0

        self.cur_x_not_fixed = set(list(range(T)))
        self.cur_x_fixed = set()


class HeuristicPricing:

    def __init__(self, data):

        (
            self.I, self.c, self.P, self.a2,
            self.T, self.D,
            self.Q_nodes_R_core, self.Q_links_R_bandwidth,
            self.q_microservices_R_core, self.q_connections_R_bandwidth,
            self.b_microservices_one, self.b_connections_one
        ) = data
        
        self.incoming = [None] * self.T
        self.outcoming = [None] * self.T
        for u in range(self.T):
            self.incoming[u] = set()
            self.outcoming[u] = set()
        for u,v in self.D:
            self.outcoming[u].add(v)
            self.incoming[v].add(u)

        self.sol = Solution(data)

        self.C_ = np.full((self.T,self.I),-1)
        self.get_feasible_col = True

        possible_microservices_on_i = []
        for i in range(self.I):
            possible_microservices_on_i.append(set())
        for u in range(self.T):
            for i in self.sol.feasible_nodes[u]:
                possible_microservices_on_i[i].add(u)
        # for i in range(self.I):
        #     print(f"possible microservices on i = {i} : {possible_microservices_on_i[i]}")
        self.possible_microservices_on_i = possible_microservices_on_i

        self.preprocessing(self.sol) # non dovrebbe fallire mai perchè esiste un mapping feasible dell'app


    def solve_multiple(self, λ_positive, μ_positive, η_n, μ = 0.2, get_feasible_col=False):

        start = perf_counter()

        I,c,a2,T,D,q_microservices_R_core,q_connections_R_bandwidth = self.I, self.c, self.a2, self.T, self.D, self.q_microservices_R_core, self.q_connections_R_bandwidth
        P = self.P
        self.get_feasible_col = get_feasible_col

        cur_sol = deepcopy(self.sol)

        #c_new = deepcopy(c)

        # for el in cur_sol.feasible_nodes:
        #     print(el)


        
        for u in range(T):

            for i in range(I):
                self.C_[u][i] = c[i]
            
            for (i, λ_value) in λ_positive:
                self.C_[u][i] += q_microservices_R_core[u] * λ_value

            if cur_sol.cur_x[u] is None:

                cur_sol.feasible_nodes[u] = sorted(cur_sol.feasible_nodes[u], key= lambda i: self.C_[u][i])
                # setting up RCL
                assert len(cur_sol.feasible_nodes[u]) > 0
                min_cost = self.C_[u][cur_sol.feasible_nodes[u][0]]
                max_cost = self.C_[u][cur_sol.feasible_nodes[u][-1]]
                t = (1-μ) * min_cost + μ * max_cost
                cur_sol.feasible_nodes[u] = [
                    i for i in cur_sol.feasible_nodes[u] if self.C_[u][i] <= t
                ]
            else:
                assert len(cur_sol.feasible_nodes[u]) == 0

        cur_sol.cur_z = 0
        for u in cur_sol.cur_x_fixed:
            cur_sol.cur_z += self.C_[u][cur_sol.cur_x[u]]

        cur_sol.cur_z -= η_n

        if not self.get_feasible_col and cur_sol.cur_z >= 0:
            return None

        err = self.preprocessing(cur_sol)
        if err:
            return None

        cur_sol.scanning_order = tuple(sorted(list(range(self.T)), key=lambda u: len(cur_sol.feasible_nodes[u])))
        
        columns = []
        appended = 0
        #runs = 0
        while True:
            for _ in range(10):
                sol, err = self.heuristic_greedy_random_try_solve2(deepcopy(cur_sol))
                #runs += 1
                if err:
                    #assert sol is None
                    continue
                #else:
                #    assert err is False

                if not self.get_feasible_col and sol.cur_z >= 0:
                    continue

                """
                sol.cur_z -= η_n

                if sol.cur_z >= 0:
                    continue
                
                sol.cur_z += sum(
                    q_microservices_R_core[u] * λ_value
                    for (i, λ_value) in λ_positive
                        for u in range(T)
                            if sol.cur_x[u] == i
                )
                # for (i, λ_value) in λ_positive:
                #     for u in range(T):
                #         if sol.cur_x[u] == i:
                #             sol.cur_z += q_microservices_R_core[u] * λ_value
                
                if sol.cur_z >= 0:
                    continue
                """

                if not self.get_feasible_col:
                    sol.cur_z += sum(
                        q_connections_R_bandwidth[u,v] * μ_value
                        for ((l,m), μ_value) in μ_positive
                            for (i,j) in a2[l][m]
                                for (u,v) in D
                                    if sol.cur_x[u] == i and sol.cur_x[v] == j
                    )

                # for ((l,m), μ_value) in μ_positive:
                #     for (i,j) in a2[l][m]:
                #         for (u,v) in D:
                #             if sol.cur_x[u] == i and sol.cur_x[v] == j:
                #                 sol.cur_z += q_connections_R_bandwidth[u,v] * μ_value

                if self.get_feasible_col or (not self.get_feasible_col and sol.cur_z < -0.001):
                    

                    # building the column
                    new_col_q_core = np.zeros(I, dtype=np.int32)
                    for u in range(T):
                        i = sol.cur_x[u]
                        new_col_q_core[i] += q_microservices_R_core[u]

                    new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
                    for (u,v) in D:
                        i = sol.cur_x[u]
                        j = sol.cur_x[v]
                        for (l,m) in P[i][j]:
                            new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

                    columns.append(
                        (
                            sum(c[sol.cur_x[u]] for u in range(T)),    # col_cost
                            new_col_q_core,                                 # col_q_core
                            new_col_q_bandwidth,                            # col_q_bandwidth 
                            deepcopy(sol.cur_x)                         # original_x
                        )
                    ) 
                    appended += 1
            
            if appended >= 10 or perf_counter()-start > 0.05:
                break

        #print("runs", runs)
        return columns if len(columns) > 0 else None


    def preprocessing(self, sol):

        T = self.T

        while True:
            found = False
            for u in range(T):
                if len(sol.feasible_nodes[u]) == 1:
                    found = True
                    # assign_microservice_u_to_unique_candidate
                    i = sol.feasible_nodes[u][0]

                    err = self.assign_microservice_u_to_node_i(sol, u, i)
                    if err: 
                        return True
                    
            if not found: 
                break
        
        return False
    
    
    def assign_microservice_u_to_node_i(self, sol, u, i):

        #c = self.c

        sol.feasible_nodes[u] = []
        sol.cur_z += self.C_[u][i]
        if not self.get_feasible_col and sol.cur_z >= 0:
           #print("not negative rc")
           return True

        sol.cur_x[u] = i
        sol.cur_x_fixed.add(u)
        sol.cur_x_not_fixed.remove(u)

        # print(f"... fixing microservice {u} on node {i}")

        # effetti collaterali del fissaggio

        ### self.update_core_consumption_after_assigning_u_to_node_i(sol, u, i)

        T = self.T
        q_microservices_R_core = self.q_microservices_R_core

        sol.residual_core[i] -= q_microservices_R_core[u]
        if sol.residual_core[i] < 0:
            #raise RuntimeError(f"FAIL ... node {i} has not enough cores to hold microservice {u}")
            return True
        
        for v in (sol.cur_x_not_fixed & self.possible_microservices_on_i[i]):
            if sol.residual_core[i] < q_microservices_R_core[v]: 
                if i in sol.feasible_nodes[v]:
                    sol.feasible_nodes[v].remove(i)
                if len(sol.feasible_nodes[v]) == 0:
                    #raise RuntimeError(f"FAIL ... elimination of node {i} from {v} candidates leaves microservice {v} with 0 candidates")
                    return True

        ### self.update_feasible_nodes_after_assigning_u_to_node_i(sol, u, i)

        b_connections_one = self.b_connections_one

        for v in (self.outcoming[u] & sol.cur_x_not_fixed):
            
            sol.feasible_nodes[v] = [
                k for k in sol.feasible_nodes[v] if (i,k) in b_connections_one[u][v]
            ]
            if len(sol.feasible_nodes[v]) == 0:
                #raise RuntimeError(f"FAIL ... elimination using b_(u,_)^(i,_) on arc (u,v) leaves microservice {v} with 0 candidates")
                return True
        
        for v in (self.incoming[u] & sol.cur_x_not_fixed):
            
            sol.feasible_nodes[v] = [
                k for k in sol.feasible_nodes[v] if (k,i) in b_connections_one[v][u]
            ]
            if len(sol.feasible_nodes[v]) == 0:
                #raise RuntimeError(f"FAIL ... elimination using b_(_,u)^(_,i) on arc (v,u) leaves microservice {v} with 0 candidates")
                return True

        ### self.update_bandwidth_consumption_after_assigning_u_to_node_i(sol, u, i)

        P = self.P
        q_connections_R_bandwidth = self.q_connections_R_bandwidth
        
        for v in (self.outcoming[u] & sol.cur_x_fixed):
            assert sol.cur_x[v] is not None
            for link in P[i][sol.cur_x[v]]:
                sol.residual_bandwidth[link] -= q_connections_R_bandwidth[u,v]
                if sol.residual_bandwidth[link] < 0:
                    #raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({u},{v})")
                    return True
                
        for v in (self.incoming[u] & sol.cur_x_fixed):
            assert sol.cur_x[v] is not None
            for link in P[sol.cur_x[v]][i]:
                sol.residual_bandwidth[link] -= q_connections_R_bandwidth[v,u]
                if sol.residual_bandwidth[link] < 0:
                    #raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({v},{u})")
                    return True
        
        return False
    
    
    def heuristic_greedy_random_try_solve2(self, sol):

        #I, T = self.I, self.T

        for u in sol.scanning_order:

            if u in sol.cur_x_not_fixed:

                l = len(sol.feasible_nodes[u])

                # if l == 0: 
                #     raise AssertionError("l == 0 missing node fixing")
                # elif l == 1: 
                #     raise AssertionError("l == 1 missing node fixing")
                # else:

                    
                #i = random.choice(sol.feasible_nodes[u])

                idx = random.choice(RNG.values[l]) #RNG.get(l)  
                i = sol.feasible_nodes[u][idx]
                
                err = self.assign_microservice_u_to_node_i(sol, u, i)
                if err:
                    return None, True
        
                err = self.preprocessing(sol)
                if err:
                    return None, True

        return sol, False