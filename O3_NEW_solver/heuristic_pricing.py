from copy import deepcopy
import math
from time import perf_counter


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
        self.d_still_to_check = D.copy()
        self.cur_x = [None] * T
        self.cur_z = 0

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
        
        self.sol = Solution(data)
        self.preprocessing(self.sol) # non dovrebbe fallire mai


    def solve(self, λ_positive, μ_positive, η_n):

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

        # for el in cur_sol.feasible_nodes:
        #     print(el)
        
        # print(λ_positive)
        # print(μ_positive)
        # print(-η_n)

        """
        - η_n
        + gp.quicksum(
            q_microservices_R_core[u] * λ_value * x[u,i] 
                for (i, λ_value) in λ_positive
                    for u in range(T)
        )
        + gp.quicksum(
            q_connections_R_bandwidth[u,v] * μ_value * x[u,i] * x[v,j] 
                for ((l,m), μ_value) in μ_positive
                        for (i,j) in a2[l][m]
                            for (u,v) in D
        )
        """

        try:
            sol_obtained = self.heuristic_ssla_try_solve(cur_sol, verbose=False)
        except Exception as e:
            #print(e)
            #print("HeuristicSingleStepLookAhead fallisce")
            return None
        
        for (i, λ_value) in λ_positive:
            for u in range(T):
                if sol_obtained.cur_x[u] == i:
                    sol_obtained.cur_z += q_microservices_R_core[u] * λ_value
        
        for ((l,m), μ_value) in μ_positive:
            for (i,j) in a2[l][m]:
                for (u,v) in D:
                    if sol_obtained.cur_x[u] == i and sol_obtained.cur_x[v] == j:
                        sol_obtained.cur_z += q_connections_R_bandwidth[u,v] * μ_value

        sol_obtained.cur_z -= η_n
        #print("HeuristicSingleStepLookAhead riesce")
        return sol_obtained
    

    def preprocessing(self, sol):

        T = self.T

        while True:
            found = False
            for u in range(T):
                if len(sol.feasible_nodes[u]) == 1:
                    found = True
                    self.assign_microservice_u_to_unique_candidate(sol, u)   
            if not found: 
                break
    

    def assign_microservice_u_to_unique_candidate(self, sol, u):

        c = self.c

        assert len(sol.feasible_nodes[u]) == 1
        i = sol.feasible_nodes[u][0]
        sol.feasible_nodes[u] = []
        sol.cur_x[u] = i
        sol.cur_z += c[i]
        # print(f"... fixing microservice {u} on node {i}")
        
        ### effetti collaterali del fissaggio
        self.update_core_consumption_after_assigning_u_to_node_i(sol, u, i)
        self.update_feasible_nodes_after_assigning_u_to_node_i(sol, u, i)
        self.update_bandwidth_consumption_after_assigning_u_to_node_i(sol, u, i)

    
    def update_core_consumption_after_assigning_u_to_node_i(self, sol, u, i):

        T = self.T
        q_microservices_R_core = self.q_microservices_R_core

        sol.residual_core[i] -= q_microservices_R_core[u]
        if sol.residual_core[i] < 0:
            raise RuntimeError(f"FAIL ... node {i} has not enough cores to hold microservice {u}")
        
        for v in range(T):
            #if len(FEASIBLE_NODES[v]) > 0 and i in FEASIBLE_NODES[v] and residual_CORE[i] < q_microservices[(v,'core')]:
            if i in sol.feasible_nodes[v] and sol.residual_core[i] < q_microservices_R_core[v]:
                sol.feasible_nodes[v].remove(i)
                #print(f"... elimination of node {i} from {v} candidates because cores are not enough")
                if len(sol.feasible_nodes[v]) == 0:
                    raise RuntimeError(f"FAIL ... elimination of node {i} from {v} candidates leaves microservice {v} with 0 candidates")


    def update_feasible_nodes_after_assigning_u_to_node_i(self, sol, u, i):

        b_connections_one = self.b_connections_one

        for arc in sol.d_still_to_check:

            if arc[0] == u and sol.cur_x[arc[1]] is None:
                v = arc[1]
                initial_len = len(sol.feasible_nodes[v])
                sol.feasible_nodes[v] = [
                    k for k in sol.feasible_nodes[v] if (i,k) in b_connections_one[u][v]
                ]
                if len(sol.feasible_nodes[v]) < initial_len:
                    #print(f"... elimination using b_(u,_)^(i,_) on arc {arc}")
                    if len(sol.feasible_nodes[v]) == 0:
                        raise RuntimeError(f"FAIL ... elimination using b_(u,_)^(i,_) on arc {arc} leaves microservice {v} with 0 candidates")
            
            elif sol.cur_x[arc[0]] is None and arc[1] == u:
                v = arc[0]
                initial_len = len(sol.feasible_nodes[v])
                sol.feasible_nodes[v] = [
                    k for k in sol.feasible_nodes[v] if (k,i) in b_connections_one[v][u]
                ]
                if len(sol.feasible_nodes[v]) < initial_len:
                    #print(f"... elimination using b_(_,u)^(_,i) on arc {arc}")
                    if len(sol.feasible_nodes[v]) == 0:
                        raise RuntimeError(f"FAIL ... elimination using b_(_,u)^(_,i) on arc {arc} leaves microservice {v} with 0 candidates")
    

    def update_bandwidth_consumption_after_assigning_u_to_node_i(self, sol, u, i):

        P = self.P
        q_connections_R_bandwidth = self.q_connections_R_bandwidth
        
        to_remove = []
        for arc in sol.d_still_to_check:
            if arc[0] == u and sol.cur_x[arc[1]] is not None:
                src_node = i
                dst_node = sol.cur_x[arc[1]]
            elif sol.cur_x[arc[0]] is not None and arc[1] == u:
                src_node = sol.cur_x[arc[0]]
                dst_node = i
            else: 
                continue 
            to_remove.append(arc)
            #print(f"... considering the bandwidth consumption on arc {arc}")
            for link in P[src_node][dst_node]:
                sol.residual_bandwidth[link] -= q_connections_R_bandwidth[arc[0],arc[1]]
                if sol.residual_bandwidth[link] < 0:
                    raise RuntimeError(f"FAIL ... link {link} has not enough bandwidth to serve the arc {arc}")
        for el in to_remove:
            sol.d_still_to_check.remove(el)

    
    def assign_microservice_u_to_cheapest_candidate(self, sol, u):

        c = self.c

        #assert len(sol.feasible_nodes[u]) == 1
        i = sol.feasible_nodes[u][0]
        sol.feasible_nodes[u] = []
        sol.cur_x[u] = i
        sol.cur_z += c[i]
        # print(f"... fixing microservice {u} on node {i}")
        
        ### effetti collaterali del fissaggio
        self.update_core_consumption_after_assigning_u_to_node_i(sol, u, i)
        self.update_feasible_nodes_after_assigning_u_to_node_i(sol, u, i)
        self.update_bandwidth_consumption_after_assigning_u_to_node_i(sol, u, i)


    def assign_microservice_u_to_candidate_at_index(self, sol, u, idx):

        c = self.c

        assert 0 <= idx < len(sol.feasible_nodes[u])
        i = sol.feasible_nodes[u][idx]
        sol.feasible_nodes[u] = []
        sol.cur_x[u] = i
        sol.cur_z += c[i]
        # print(f"... fixing microservice {u} on node {i}")
        
        ### effetti collaterali del fissaggio
        self.update_core_consumption_after_assigning_u_to_node_i(sol, u, i)
        self.update_feasible_nodes_after_assigning_u_to_node_i(sol, u, i)
        self.update_bandwidth_consumption_after_assigning_u_to_node_i(sol, u, i)
    

    def heuristic_greedy_try_solve(self, sol, verbose=True):
        
        start = perf_counter()

        I, T = self.I, self.T

        while True:
            found = False

            min_candidates = I
            min_u = None

            for u in range(T):
                l = len(sol.feasible_nodes[u])
                if l == 0:
                    assert sol.cur_x[u] is not None
                    continue
                #elif l == 1:
                    # found = True
                    # min_u = None
                    # print(f"... fixing microservice {u} because it has only one candidate\n")
                    # CUR_Z = fix_microservice(FEASIBLE_NODES, u, CUR_Z, CUR_X, d, residual_CORE, residual_BANDWIDTH)
                    # break
                else:
                    if l == 1:
                        raise RuntimeError("missing node fixing")
                    # l > 1
                    found = True
                    if l < min_candidates:
                        min_candidates = l
                        min_u = u

            if not found: 
                # se tutti i nodi sono stati fissati allora ho finito con successo
                break
            
            assert min_u is not None
            self.assign_microservice_u_to_cheapest_candidate(sol, min_u)
            self.preprocessing(sol)
            #print(f"... fixing microservice {min_u} because it has the minimum number of candidates\n")
            #CUR_Z = fix_microservice(FEASIBLE_NODES, min_u, CUR_Z, CUR_X, d, residual_CORE, residual_BANDWIDTH)

        end = perf_counter()
        time = end-start

        assert len(sol.d_still_to_check) == 0

        if verbose:
            print("\n!!! SUCCESS - HeuristicGreedy")
            print(f"execution time = {time}")
            print(f"z_heuristic = {sol.cur_z}")
            #for u in T:
            #    print(f"microservice {u:2d} on node {self.sol.cur_x[u]}")
        
        return sol

    
    def heuristic_ssla_try_solve(self, sol, verbose=True):

        start = perf_counter()

        I, T = self.I, self.T

        while True:
            found = False

            min_candidates = I
            min_u = None

            for u in range(T):
                l = len(sol.feasible_nodes[u])
                if l == 0:
                    assert sol.cur_x[u] is not None
                    continue
                #elif l == 1:
                    # found = True
                    # min_u = None
                    # print(f"... fixing microservice {u} because it has only one candidate\n")
                    # CUR_Z = fix_microservice(FEASIBLE_NODES, u, CUR_Z, CUR_X, d, residual_CORE, residual_BANDWIDTH)
                    # break
                else:
                    if l == 1:
                        raise RuntimeError("missing node fixing")
                    # l > 1
                    found = True
                    if l < min_candidates:
                        min_candidates = l
                        min_u = u

            if not found: 
                # se tutti i nodi sono stati fissati allora ho finito con successo
                break
            
            assert min_u is not None

            best_z = math.inf
            best_idx = None 
            for idx, node in enumerate(sol.feasible_nodes[min_u]):
                try:
                    #print(f"trying to fix {min_u} on {node}")
                    sol_copy = deepcopy(sol)
                    self.assign_microservice_u_to_candidate_at_index(sol_copy, min_u, idx)
                    self.preprocessing(sol_copy)
                    z = self.heuristic_greedy_try_solve(deepcopy(sol_copy), verbose=False).cur_z
                except Exception as e:
                    #print("exception occurred")
                    continue
                if z < best_z:
                    best_z = z
                    best_idx = idx

            if best_idx is None:
                raise RuntimeError("no fixing available") 
            
            self.assign_microservice_u_to_candidate_at_index(sol, min_u, best_idx)
            self.preprocessing(sol)

        end = perf_counter()
        time = end-start

        assert len(sol.d_still_to_check) == 0

        if verbose:
            print("\n!!! SUCCESS - HeuristicSingleStepLookAhead")
            print(f"execution time = {time}")
            print(f"z_heuristic = {sol.cur_z}")
            #for u in T:
            #    print(f"microservice {u:2d} on node {self.sol.cur_x[u]}")
        
        return sol