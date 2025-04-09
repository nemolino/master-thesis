from copy import deepcopy
import numpy as np
from time import perf_counter
import random
random.seed(42)

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

    """
    def __str__(self):
        s = f"current z = {self.cur_z}\n"
        s += f"current x = {self.cur_x}\n"
        for u in range(INS.T):
            if len(self.feasible_nodes[u]) > 0:
                s += f"microservice {u:2d} can be placed in {len(self.feasible_nodes[u]):2d}/{INS.I:2d} nodes : {self.feasible_nodes[u]}\n"
        return s
    

    def as_str_long(self):
        s = f"current z = {self.cur_z}\n"
        s += f"current x = {self.cur_x}\n"
        for u in range(INS.T):
            if len(self.feasible_nodes[u]) > 0:
                s += f"microservice {u:2d} can be placed in {len(self.feasible_nodes[u]):2d}/{INS.I:2d} nodes : {self.feasible_nodes[u]}\n"
        s += f"{self.residual_core}\n"
        s += f"{self.residual_bandwidth}\n"
        s += f"{self.d_still_to_check}\n"
        return s
    """


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


    def generate_feasible_columns_semigreedy(self):

        cur_sol = deepcopy(self.sol)

        columns = []

        # μ tunes the size of RCL
        # μ = 0
        sol, err = self.heuristic_greedy_try_solve(deepcopy(cur_sol))
        if err:
            assert sol is None
        else:
            assert err is False
            columns.append(sol)

        # try:
        #     sol = self.heuristic_greedy_random_try_solve(deepcopy(cur_sol))
        #     columns.append(sol)
        # except Exception as e:
        #     pass

        for μ in [0.2, 0.4, 0.6, 0.8, 1.0]:

            print(f"μ = {μ}", end=' ')
            
            cur_sol_μ = deepcopy(cur_sol)
            for u in range(self.T):
                if cur_sol_μ.cur_x[u] is None:
                    assert len(cur_sol_μ.feasible_nodes[u]) > 0
                    min_cost = self.c[cur_sol_μ.feasible_nodes[u][0]]
                    max_cost = self.c[cur_sol_μ.feasible_nodes[u][-1]]
                    t = (1-μ) * min_cost + μ * max_cost
                    cur_sol_μ.feasible_nodes[u] = [
                        i for i in cur_sol_μ.feasible_nodes[u] if self.c[i] <= t
                    ]

            err = self.preprocessing(cur_sol_μ)
            if err:
                print(f"infeasible")
                continue

            # for u in range(self.T):
            #     print(f"{u} : {cur_sol_μ.feasible_nodes[u]}")
            
            runs_count = 0
            total_time = 0
            appended = 0

            limit = 10
            for _ in range(1000):
                # try:
                #     sol = self.heuristic_greedy_random_try_solve(deepcopy(cur_sol_μ))
                # except Exception as e:
                #     continue
                s = perf_counter()
                sol, err = self.heuristic_greedy_random_try_solve(deepcopy(cur_sol_μ))
                e = perf_counter()
                total_time += e-s
                runs_count +=1 
                if err:
                    assert sol is None
                    continue
                else:
                    assert err is False

                columns.append(sol)
                appended += 1
                if appended >= limit: 
                    break

            print(f"generated {appended} columns")
            print(f"runs_count {runs_count} , total time {total_time} , avg time x run {total_time/runs_count}")
        
        return columns
        
        """
        columns = sorted(columns, key=lambda x: x.cur_z)
        low_cost_columns = columns[:30]
        
        M = np.zeros((self.T, self.I), dtype=np.int32)
        for col in columns:
            for u in range(self.T):
                M[u,col.cur_x[u]] += 1

        diversification_columns = sorted(columns[30:], key=lambda col: sum(M[u,col.cur_x[u]] for u in range(self.T)))[:70]
        
        return low_cost_columns + diversification_columns
        """

    
    def solve_multiple(self, λ_positive, μ_positive, η_n):

        I,c,a2,T,D,q_microservices_R_core,q_connections_R_bandwidth = self.I, self.c, self.a2, self.T, self.D, self.q_microservices_R_core, self.q_connections_R_bandwidth

        cur_sol = deepcopy(self.sol)

        c_new = deepcopy(c)

        # for el in cur_sol.feasible_nodes:
        #     print(el)

        for u in range(T):
            for i in range(I):
                c_new[i] = c[i]
            for (i, λ_value) in λ_positive:
                c_new[i] += q_microservices_R_core[u] * λ_value
            cur_sol.feasible_nodes[u] = sorted(cur_sol.feasible_nodes[u], key= lambda i: c_new[i])


        appended = 0
        columns = []
        for μ in [0.2]:

            cur_sol_μ = deepcopy(cur_sol)

            for u in range(self.T):
                if cur_sol_μ.cur_x[u] is None:
                    assert len(cur_sol_μ.feasible_nodes[u]) > 0
                    min_cost = c_new[cur_sol_μ.feasible_nodes[u][0]]
                    max_cost = c_new[cur_sol_μ.feasible_nodes[u][-1]]
                    t = (1-μ) * min_cost + μ * max_cost
                    cur_sol_μ.feasible_nodes[u] = [
                        i for i in cur_sol_μ.feasible_nodes[u] if c_new[i] <= t
                    ]
            # for u in range(self.T):
            #     print(f"{u} : {cur_sol_μ.feasible_nodes[u]}")

            err = self.preprocessing(cur_sol_μ)
            if err:
                continue

            k = 100
            for _ in range(k):

                sol, err = self.heuristic_greedy_random_try_solve(deepcopy(cur_sol_μ))
                if err:
                    assert sol is None
                    continue
                else:
                    assert err is False

                sol.cur_z -= η_n

                if sol.cur_z >= 0:
                    continue

                for (i, λ_value) in λ_positive:
                    for u in range(T):
                        if sol.cur_x[u] == i:
                            sol.cur_z += q_microservices_R_core[u] * λ_value
                
                for ((l,m), μ_value) in μ_positive:
                    for (i,j) in a2[l][m]:
                        for (u,v) in D:
                            if sol.cur_x[u] == i and sol.cur_x[v] == j:
                                sol.cur_z += q_connections_R_bandwidth[u,v] * μ_value

                if sol.cur_z < -0.001:
                    columns.append(sol)
                    appended += 1
                    if appended >= 5:
                        break

        return columns if len(columns) > 0 else None

        # elif len(sols_obtained) <= how_many_max:
        #     return sols_obtained
        # else:
        #     # get the most diversified how_many_max columns

        #     M = np.zeros((T, I), dtype=np.int32)
        #     for sol in sols_obtained:
        #         for u in range(T):
        #             M[u][sol.cur_x[u]] += 1
        #     print(M)
        #     return sols_obtained[:how_many_max]

            
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

        c = self.c

        sol.feasible_nodes[u] = []
        sol.cur_x[u] = i
        sol.cur_z += c[i]
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
    
    
    # def assign_microservice_u_to_candidate_at_index(self, sol, u, idx):

    #     assert 0 <= idx < len(sol.feasible_nodes[u])
    #     i = sol.feasible_nodes[u][idx]
    #     self.assign_microservice_u_to_node_i(sol, u, i)

    """
    # iterazione
    # rimozione dal sol.feasible_nodes[x]
    # prendi il meno costoso, random o uno specifico
    def update_core_consumption_after_assigning_u_to_node_i(self, sol, u, i):

        T = self.T
        q_microservices_R_core = self.q_microservices_R_core

        sol.residual_core[i] -= q_microservices_R_core[u]
        if sol.residual_core[i] < 0:
            raise RuntimeError(f"FAIL ... node {i} has not enough cores to hold microservice {u}")
        
        for v in (sol.cur_x_not_fixed & self.possible_microservices_on_i[i]):
            if sol.residual_core[i] < q_microservices_R_core[v]: 
                if i in sol.feasible_nodes[v]:
                    sol.feasible_nodes[v].remove(i)
                if len(sol.feasible_nodes[v]) == 0:
                    raise RuntimeError(f"FAIL ... elimination of node {i} from {v} candidates leaves microservice {v} with 0 candidates")
    


    # per ogni u devo avere tutti gli archi (u,v) e gli archi (v,u) con v non ancora fissato
    # rimozione di multipli elementi
    def update_feasible_nodes_after_assigning_u_to_node_i(self, sol, u, i):

        b_connections_one = self.b_connections_one

        for v in (self.outcoming[u] & sol.cur_x_not_fixed):
            
            sol.feasible_nodes[v] = [
                k for k in sol.feasible_nodes[v] if (i,k) in b_connections_one[u][v]
            ]
            if len(sol.feasible_nodes[v]) == 0:
                raise RuntimeError(f"FAIL ... elimination using b_(u,_)^(i,_) on arc (u,v) leaves microservice {v} with 0 candidates")
        
        for v in (self.incoming[u] & sol.cur_x_not_fixed):
            
            sol.feasible_nodes[v] = [
                k for k in sol.feasible_nodes[v] if (k,i) in b_connections_one[v][u]
            ]
            if len(sol.feasible_nodes[v]) == 0:
                raise RuntimeError(f"FAIL ... elimination using b_(_,u)^(_,i) on arc (v,u) leaves microservice {v} with 0 candidates")
    
    
    # per ogni u devo avere tutti gli archi (u,v) e gli archi (v,u) con v già fissato
    def update_bandwidth_consumption_after_assigning_u_to_node_i(self, sol, u, i):

        P = self.P
        q_connections_R_bandwidth = self.q_connections_R_bandwidth
        
        for v in (self.outcoming[u] & sol.cur_x_fixed):
            assert sol.cur_x[v] is not None
            for link in P[i][sol.cur_x[v]]:
                sol.residual_bandwidth[link] -= q_connections_R_bandwidth[u,v]
                if sol.residual_bandwidth[link] < 0:
                    raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({u},{v})")
                
        for v in (self.incoming[u] & sol.cur_x_fixed):
            assert sol.cur_x[v] is not None
            for link in P[sol.cur_x[v]][i]:
                sol.residual_bandwidth[link] -= q_connections_R_bandwidth[v,u]
                if sol.residual_bandwidth[link] < 0:
                    raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc ({v},{u})")
    """
    

    def heuristic_greedy_try_solve(self, sol):
        
        I, T = self.I, self.T

        while len(sol.cur_x_not_fixed) > 0:

            min_candidates = I+1
            min_u = None

            for u in sol.cur_x_not_fixed:
                l = len(sol.feasible_nodes[u])
                
                if l == 0: raise AssertionError("l == 0 missing node fixing")
                if l == 1: raise AssertionError("l == 1 missing node fixing")
                if l < min_candidates:
                    assert l >= 2
                    min_candidates = l
                    min_u = u

            assert min_u is not None
            
            # assign_microservice_u_to_cheapest_candidate
            u = min_u
            i = sol.feasible_nodes[u][0]

            err = self.assign_microservice_u_to_node_i(sol, u, i)
            if err:
                return None, True
            
            err = self.preprocessing(sol)
            if err:
                return None, True
        
        return sol, False
    
    
    def heuristic_greedy_random_try_solve(self, sol):

        I, T = self.I, self.T
        
        while len(sol.cur_x_not_fixed) > 0:
            
            min_candidates = I+1
            min_u = None

            for u in sol.cur_x_not_fixed:
                l = len(sol.feasible_nodes[u])

                if l == 0: raise AssertionError("l == 0 missing node fixing")
                if l == 1: raise AssertionError("l == 1 missing node fixing")
                if l < min_candidates:
                    assert l >= 2
                    min_candidates = l
                    min_u = u

            assert min_u is not None

            # assign_microservice_u_to_random_candidate
            u = min_u
            i = random.choice(sol.feasible_nodes[u])

            #self.assign_microservice_u_to_node_i(sol, u, i)
            #self.preprocessing(sol)

            err = self.assign_microservice_u_to_node_i(sol, u, i)
            if err:
                return None, True
            
            err = self.preprocessing(sol)
            if err:
                return None, True

        return sol, False