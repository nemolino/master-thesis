from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
import numpy as np
from copy import deepcopy

from instance import Instance, InfeasibleOnBuilding
from math import ceil

options = {}
with open("../gurobi.lic") as f:
    lines = f.readlines()
    options["WLSACCESSID"] = lines[3].strip().split('=')[1]
    options["WLSSECRET"] = lines[4].strip().split('=')[1]
    options["LICENSEID"] = int(lines[5].strip().split('=')[1])
    options["OutputFlag"] = 0


class InfeasiblePricing(Exception):

    def __init__(self, message="InfeasiblePricing"):
        self.message = message
        super().__init__(self.message)


class InfeasibleMaster(Exception):

    def __init__(self, message="InfeasibleMaster"):
        self.message = message
        super().__init__(self.message)


class Column:

    def __init__(self, col_cost, col_q_core, col_q_bandwidth, original_x):
        self.col_cost = col_cost
        self.col_q_core = col_q_core
        self.col_q_bandwidth = col_q_bandwidth
        self.original_x = original_x

    def __eq__(self, other):
        assert self.col_q_core.shape == other.col_q_core.shape
        assert self.col_q_bandwidth.shape == other.col_q_bandwidth.shape
        return (
            self.col_cost == other.col_cost and
            (self.col_q_core == other.col_q_core).all() and 
            (self.col_q_bandwidth == other.col_q_bandwidth).all()
        )


class Master:

    def __init__(self, data, dummy_columns_costs, number_of_microservices, logger=None):

        self.data = data

        self.COLUMNS_POOL = None
        self.logger = logger
        self.z = None
        self.model = None
        self.initialize_columns_pool(dummy_columns_costs, number_of_microservices)
        self.build()
    
    
    def initialize_columns_pool(self, dummy_columns_costs, number_of_microservices):

        assert self.COLUMNS_POOL is None
        
        (
            I,A,N,_,_
        ) = self.data

        COLUMNS_POOL = []
        for _ in range(N): 
            COLUMNS_POOL.append([])

        for n in range(N): 

            dummy_col_cost = dummy_columns_costs[n]
            dummy_col_q_core = np.zeros(I, dtype=np.int32)
            """
            dummy_col_q_bandwidth = np.full((I, I), -1, dtype=np.int32)
            for (l,m) in A:
                dummy_col_q_bandwidth[l,m] = 0
            """
            dummy_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
            dummy_col = Column(
                dummy_col_cost, dummy_col_q_core, dummy_col_q_bandwidth, [None] * number_of_microservices[n]
            )

            COLUMNS_POOL[n].append(dummy_col)

        self.COLUMNS_POOL = COLUMNS_POOL

    
    def add_column(self, n, new_column, checked=True):

        assert self.COLUMNS_POOL    is not None
        assert self.logger          is not None
        assert self.model           is not None
        assert self.z               is not None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        #LOG = self.logger
        model = self.model
        z = self.z 
        
        if checked and new_column in self.COLUMNS_POOL[n]:
            raise RuntimeError(f"Master add_column : added duplicate column for {n}-th app")
    
        self.COLUMNS_POOL[n].append(new_column)

        # update master
        #start = perf_counter()
        z[n].append(model.addVar(vtype=GRB.CONTINUOUS, lb=0.0))
        p = len(z[n])-1

        z[n][p].Obj = new_column.col_cost

        for i in range(I):
            model.chgCoeff(
                model.getConstrByName(f"on_node_{i}_consumption_of_core"), 
                z[n][p], 
                -new_column.col_q_core[i] # -COLUMNS_POOL[n][-1].col_q_core[i]
            )

        for (l,m) in A:
            model.chgCoeff(
                model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth"), 
                z[n][p], 
                -new_column.col_q_bandwidth[l,m]
            )

        model.chgCoeff(
            model.getConstrByName(f"convexity_{n}"), 
            z[n][p], 
            1
        )
        #end = perf_counter()
        # LOG.info(f"... master update in {end-start:.3f} s")
     
    
    def build(self):

        assert self.COLUMNS_POOL    is not None
        assert self.logger          is not None
        assert self.z               is None
        assert self.model           is None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        #LOG = self.logger

        #start = perf_counter()

        # build and optimize the model

        COLUMNS_POOL = self.COLUMNS_POOL

        model = gp.Model("master", env=gp.Env(params=options))

        #for n in range(N):  
        #    print(f"n {n} has len {len(COLUMNS_POOL[n])}")

        #variable_type = GRB.BINARY if discrete else GRB.CONTINUOUS
        z = [[None] * (len(COLUMNS_POOL[n])) for n in range(N)]
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
                
        
        # print('\033[91m' + f"... building and solving master" + '\033[0m' + f"   , {len(z)} columns available\n")
        # LOG.info(f"... building and solving master   , {sum(len(COLUMNS_POOL[n]) for n in range(N))} columns available")
        # LOG.info(f"... building master   , {sum(len(COLUMNS_POOL[n]) for n in range(N))} columns available")
        # LOG.info("")

        model.setObjective(
            sum(
                COLUMNS_POOL[n][p].col_cost * z[n][p]
                for n in range(N) 
                    for p in range(len(COLUMNS_POOL[n]))
            ), 
            GRB.MINIMIZE
        )

        for i in range(I):
            model.addConstr(
                - sum(
                    COLUMNS_POOL[n][p].col_q_core[i] * z[n][p]
                    for n in range(N) 
                        for p in range(len(COLUMNS_POOL[n]))
                ) >= - Q_nodes_R_core[i], 
                name = f"on_node_{i}_consumption_of_core"
            )

        for (l,m) in A:
            model.addConstr(
                - sum(
                    COLUMNS_POOL[n][p].col_q_bandwidth[l,m] * z[n][p]
                    for n in range(N) 
                        for p in range(len(COLUMNS_POOL[n]))
                ) >= - Q_links_R_bandwidth[l,m],
                name = f"on_link_({l},{m})_consumption_of_bandwidth"
            )

        for n in range(N):
            model.addConstr(
                    sum(z[n][p] for p in range(len(COLUMNS_POOL[n]))) >= 1,
                    name = f"convexity_{n}"
                )
        
        model.update()
        # end = perf_counter()

        # LOG.info(f"master build in {end-start:.3f} s")
        # LOG.info("")

        self.z = z
        self.model = model

    
    def optimize(self):

        assert self.COLUMNS_POOL    is not None
        assert self.logger          is not None
        assert self.z               is not None
        assert self.model           is not None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        COLUMNS_POOL = self.COLUMNS_POOL
        #LOG = self.logger
        z = self.z
        model = self.model

        # LOG.info(f"... solving master   , {sum(len(COLUMNS_POOL[n]) for n in range(N))} columns available")
        # LOG.info("")

        for n in (range(N)): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.CONTINUOUS
                z[n][p].lb = 0.0

        model.optimize()

        if model.Status != GRB.OPTIMAL and model.Status != GRB.INFEASIBLE:
            raise RuntimeError("CG Master optimize : master status diverso da optimal e da infeasible")
        
        if model.Status == GRB.INFEASIBLE:
            raise InfeasibleMaster()

        #print(f"z* = {model.ObjVal} , master solve in {model.Runtime:.3f}\n")
        # LOG.info(f"z* = {model.ObjVal} , master solve in {model.Runtime:.3f}")
        # LOG.info("")

        #for n in range(N):
            #print(f"application {n}, {len(COLUMNS_POOL[n])} patterns : {[round(z[n][p].X,2) for p in range(len(COLUMNS_POOL[n]))]}")
            # LOG.debug(f"application {n}, {len(COLUMNS_POOL[n])} patterns : {[round(z[n][p].X,2) for p in range(len(COLUMNS_POOL[n]))]}")
        #print()
        # LOG.debug("")

        
        #start = perf_counter()

        # getting dual values
        epsilon = 0.001

        λ_positive = []
        for i in range(I):
            value = model.getConstrByName(f"on_node_{i}_consumption_of_core").Pi
            assert value >= -epsilon
            if value > 0:
                λ_positive.append((i,value))
        λ_positive = tuple(λ_positive)


        μ_positive = []
        for (l,m) in A:
            value = model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").Pi
            assert value >= -epsilon
            if value > 0:
                μ_positive.append(((l,m),value))
        μ_positive = tuple(μ_positive)
            
        η = []
        for n in range(N):
            value = model.getConstrByName(f"convexity_{n}").Pi
            assert value >= -epsilon
            η.append(value)
        η = tuple(η)

        #end = perf_counter()
        # print(f"dual values retrieval in {end-start:.3f} s\n")
        # LOG.info(f"dual values retrieval in {end-start:.3f} s")
        # LOG.info("")

        return λ_positive, μ_positive, η, model.ObjVal
    
    
    def is_infeasible(self, still_to_fix=None):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data
        
        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        z = self.z

        epsilon = 0.001
        for n in (range(N) if still_to_fix is None else still_to_fix): 
            if z[n][0].X > epsilon:
                return True
        return False

    
    def solution_is_integer(self, N):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        assert self.logger is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z

        for n in range(N): 
            found = False
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X >= 0.999:
                    found = True
                    break
            if not found:
                return False

        return True
    

class Pricing:


    def __init__(self, data, logger=None):

        self.data = data
        self.logger = logger
        self.model = None
        self.x = None
        self.build()


    def build(self):

        assert self.logger is not None
        assert self.model is None
        assert self.x is None

        (
            I,A,c,P,a2,
            T,D,
            Q_nodes_R_core, Q_links_R_bandwidth,
            q_microservices_R_core, q_connections_R_bandwidth,
            b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual
        ) = self.data

        model = gp.Model("pricing", env=gp.Env(params=options))

        x = model.addVars(T, I, vtype=GRB.BINARY, name = "x")

        #start = perf_counter()
        model.addConstrs(
            (
                gp.quicksum(x[u,i] for i in range(I)) == 1 
                for u in range(T)
            ), 
            name='microservice_u_mapped_to_exactly_one_node'
        )
        #end = perf_counter()
        #print(f"{end-start:.3f} s to build 'microservice_u_mapped_to_exactly_one_node' constraints")

        #start = perf_counter()
        model.addConstrs(
            (
                x[u,i] == 0 
                for u in range(T) 
                    for i in b_microservices_zero[u]
            ), 
            name='microservice_u_not_mappable_to_node_i'
        )
        #end = perf_counter()
        #print(f"{end-start:.3f} s to build 'microservice_u_not_mappable_to_node_i' constraints")

        #start = perf_counter()
        model.addConstrs(
            (
                x[u,i] + x[v,j] <= 1
                for (u,v) in D
                    for (i,j) in b_connections_zero_not_implied[u][v]
            ),
            name='microservices_(u,v)_not_mappable_to_nodes_(i,j)'
        )
        #end = perf_counter()
        #print(f"{end-start:.3f} s to build 'microservices_(u,v)_not_mappable_to_nodes_(i,j)' constraints")

        #start = perf_counter()
        self.consumption_core_constraint = model.addConstrs(
            (
                gp.quicksum(q_microservices_R_core[u] * x[u,i] for u in range(T)) <= Q_nodes_R_core[i]
                for i in range(I)
            ),
            name='on_node_i_consumption_of_resource_core'
        )
        #end = perf_counter()
        #print(f"{end-start:.3f} s to build 'on_node_i_consumption_of_resource_core' constraints")
        
        #start = perf_counter()
        lhs = dict()
        for (l,m) in A:
            lhs[(l,m)] = gp.QuadExpr(0)
            
        for (u,v) in D:
            for (i,j) in b_connections_one_actual[u][v]:
                for (l,m) in P[i][j]:
                    lhs[(l,m)].add(q_connections_R_bandwidth[u,v] * x[u,i] * x[v,j])
    
        self.consumption_bandwidth_constraint = model.addConstrs(
            (
                lhs[(l,m)] <= Q_links_R_bandwidth[l,m]
                for (l,m) in A
            ),
            name='on_link_(l,m)_consumption_of_resource_bandwidth'
        )
        #end = perf_counter()
        #print(f"{end-start:.3f} s to build 'on_link_(l,m)_consumption_of_resource_bandwidth' constraints")
        
        self.model = model
        self.x = x


    def optimize(self, λ_positive, μ_positive, η_n):

        assert self.logger is not None
        assert self.model is not None
        assert self.x is not None

        model = self.model
        x = self.x 
        #LOG = self.logger

        (
            I,A,c,P,a2,
            T,D,
            Q_nodes_R_core, Q_links_R_bandwidth,
            q_microservices_R_core, q_connections_R_bandwidth,
            b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual
        ) = self.data

        #start = perf_counter()
        model.setObjective(
            gp.quicksum(c[i] * x[u,i] for u in range(T) for i in range(I)) 
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
            ), 
            GRB.MINIMIZE
        )
        #end = perf_counter()
        #obj_build_time = end-start

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise InfeasiblePricing()
        
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("ERROR : pricing status diverso da optimal")

        # >= -epsilon oppure >= 0
        # epsilon = 0.001
        # if model.ObjVal >= -epsilon:
        #    return None, None
        
        #start = perf_counter()

        optimal_assignment = [None] * T
        for u in range(T):
            for i in range(I):
                if x[u,i].X > 0.5: # ! this is the preferred way to check if a binary variable is 1
                    assert optimal_assignment[u] is None
                    optimal_assignment[u] = i
            assert optimal_assignment[u] is not None

        #new_column = defaultdict(lambda: None)
        """
        new_column["cost"] = sum(c[i] * x[u,i].X 
                                 for u in T 
                                 for i in I)
        """
        new_col_cost = sum(c[optimal_assignment[u]] for u in range(T))
        
        """
        for i in I:
            for k in R:
                new_column[(i,k)] = 0
        for u in T:
            i = optimal_assignment[u]
            for k in R:
                new_column[(i,k)] += q_microservices[(u,k)]
        """

        new_col_q_core = np.zeros(I, dtype=np.int32)
        for u in range(T):
            i = optimal_assignment[u]
            new_col_q_core[i] += q_microservices_R_core[u]

        """
        for (l,m) in A:
            for k in R:
                new_column[((l,m),k)] = 0

        for (u,v) in D:
            i = optimal_assignment[u]
            j = optimal_assignment[v]
            for (l,m) in a_format3[(i,j)]:
                for k in R:
                    new_column[((l,m),k)] += q_connections[((u,v),k)] 
        """

        new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
        for (u,v) in D:
            i = optimal_assignment[u]
            j = optimal_assignment[v]
            for (l,m) in P[i][j]:
                new_col_q_bandwidth[l,m] += q_connections_R_bandwidth[u,v]

        new_col = Column(
            new_col_cost, new_col_q_core, new_col_q_bandwidth, optimal_assignment
        )

        # end = perf_counter()
        # column_build_time = end-start
        
        # print(f"c* = {model.ObjVal:12.6f} , obj build in {obj_build_time:.3f}s , solve in {model.Runtime:.3f}s , #nodes = {model.NodeCount} , col build in {column_build_time:.3f}s")
        # LOG.info(f"c* = {model.ObjVal:12.6f} , obj build in {obj_build_time:.3f}s , solve in {model.Runtime:.3f}s , #nodes = {model.NodeCount} , col build in {column_build_time:.3f}s")

        return new_col, x, model.ObjVal
    
    

class ColumnGeneration:

    
    def __init__(self, N=None, filenames=None, logger=None):

        self.N = N                      # number of apps
        self.filenames = filenames
        self.logger = logger


    def execute(self):

        assert self.N is not None and self.N > 0
        assert self.filenames is not None
        assert self.logger is not None

        N = self.N
        network_filename, network_rp_filename, app_filenames, app_rp_filenames = self.filenames
        LOG = self.logger

        start_CG = perf_counter()

        # --- Building data structures ----------------------------------------

        I,A,c,P,a2 = Instance.build_network_structure_from_file(network_filename)

        (
            Q_nodes_R_core, 
            Q_nodes_S_has_camera, 
            Q_nodes_S_has_gpu, 
            Q_links_R_bandwidth, 
            Q_links_S_latency 
        ) = Instance.build_network_rp_availability_from_file(network_rp_filename, data=(I,A))

        max_node_cost = ceil(np.max(c))+1

        ### Building micro-service structure for each application (needed for pricings)

        APP = [None] * N

        for n in range(N):

            APP[n] = dict()

            ### build_microservices_structure_from_file
            
            T,D = Instance.build_app_structure_from_file(app_filenames[n])
            
            APP[n]["T"] = T
            APP[n]["D"] = D

            ### build_microservices_properties_resources_consumption_from_file

            (
                q_microservices_R_core, 
                q_microservices_S_has_camera, 
                q_microservices_S_has_gpu, 
                q_connections_R_bandwidth, 
                q_connections_S_latency
            ) = Instance.build_app_rp_consumption_from_file(
                app_rp_filenames[n], data=(T,D)
            )
            
            APP[n]["q_microservices_R_core"] = q_microservices_R_core
            APP[n]["q_connections_R_bandwidth"] = q_connections_R_bandwidth

            ### build_b_coefficients
            try:
                (
                    b_microservices_zero, 
                    b_microservices_one,
                    b_connections_zero_not_implied, 
                    b_connections_one, b_connections_one_actual
                ) = Instance.build_b_coefficients(
                    data= (
                        I, T, D, P,
                        Q_nodes_R_core, Q_links_R_bandwidth, 
                        q_microservices_R_core, q_connections_R_bandwidth,
                        Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
                        q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_S_latency
                    )
                )
            except InfeasibleOnBuilding:
                print("INFEASIBLE INSTANCE : (on building) there is a microservice you cannot place in any node")
                # LOG.warning(f"INFEASIBLE INSTANCE : (on building) there is a microservice you cannot place in any node")
                return None, None, "infeasible building"
                
            APP[n]["b_microservices_zero"] = b_microservices_zero
            APP[n]["b_microservices_one"] = b_microservices_one
            APP[n]["b_connections_zero_not_implied"] = b_connections_zero_not_implied
            APP[n]["b_connections_one"] = b_connections_one
            APP[n]["b_connections_one_actual"] = b_connections_one_actual

        # --- Building CG structures ----------------------------------------------------

        dummy_columns_costs = [APP[n]["T"] * max_node_cost for n in range(N)]
        number_of_microservices = [APP[n]["T"] for n in range(N)]

        master = Master(
            (I,A,N,Q_nodes_R_core,Q_links_R_bandwidth), 
            dummy_columns_costs, number_of_microservices,
            logger = LOG
        )

        pricings = []

        for n in range(N):

            pricings.append(
                Pricing(
                    data = (
                        I,A,c,P,a2,
                        APP[n]["T"], APP[n]["D"],
                        Q_nodes_R_core, Q_links_R_bandwidth,
                        APP[n]["q_microservices_R_core"], 
                        APP[n]["q_connections_R_bandwidth"],
                        APP[n]["b_microservices_zero"], 
                        APP[n]["b_connections_zero_not_implied"],
                        APP[n]["b_connections_one_actual"]
                    ),
                    logger = LOG
                )
            )
            # LOG.info(f"{n}-th pricing build")
            
            # adding column of the optimal mapping of n-th application
            try:
                first_column, _, _ = pricings[n].optimize(
                    λ_positive=(), μ_positive=(), η_n=0
                )
            except InfeasiblePricing:
                print(f"INFEASIBLE INSTANCE : {n}-th pricing infeasible with dual values = 0, meaning that you cannot map {n}-th application")
                # LOG.warning(f"INFEASIBLE INSTANCE : {n}-th pricing infeasible with dual values = 0, meaning that you cannot map {n}-th application")
                return None, None, "infeasible pricing"

            master.add_column(n, first_column)
            # print(f"{n}-th pricing solved with dual values = 0, the column is the optimal mapping of {n}-th application\n")
            # LOG.info(f"{n}-th pricing solved with dual values = 0, the column is the optimal mapping of {n}-th application")
            # LOG.info("")
        

        
        # print("*" * 80 + "\n")
        # LOG.info("*" * 80)
        # LOG.info("")

        # --- CG ------------------------------------------------------------------------

        EPSILON = 0.001
        STOP_GAP = 0.1

        #primal_bound = None
        iterations_count = 0
        best_dual_bound = 0

        while True:

            iterations_count += 1
            
            #print(f"iteration {iterations_count}\n")
            
            # non deve essere infeasible perchè ho le dummy
            λ_positive, μ_positive, η, master_obj_val = master.optimize() 

            columns_to_add = []
            for n in range(N):
                columns_to_add.append([])

            sum_reduced_costs = 0
            new_columns_found = False
            
            for n in range(N):

                #print('\033[91m' + f"... solving pricing of {n}-th application\t\t" + '\033[0m', end='')

                # non deve essere infeasible
                new_column, x, reduced_cost = pricings[n].optimize(λ_positive, μ_positive, η[n])
                
                if reduced_cost >= -EPSILON:
                    #print("--> NOT added to master")
                    pass
                else:
                    #print("--> added to master")
                    sum_reduced_costs += reduced_cost
                    new_columns_found = True
                    columns_to_add[n].append(new_column)
                    assert len(columns_to_add[n]) == 1
                
            dual_bound = master_obj_val + sum_reduced_costs
                
            best_dual_bound = max(best_dual_bound, dual_bound)
            gap = master_obj_val-best_dual_bound
            perc_gap = 100 * gap / master_obj_val
                
            # print(f"\n cur dual bound : {dual_bound:.5f}") 
            # print(f"best dual bound : {best_dual_bound:.5f}")
            # print(f"            gap : {gap:.5f} ( {perc_gap:.6f} % )")

            if perc_gap <= STOP_GAP and not master.is_infeasible():   
                end_CG = perf_counter()
                print(f"\nSTOP, gap < {STOP_GAP} %")
                print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                ### controllo se l'ottimo è già intero
                if master.solution_is_integer(N):
                    print(f"CG_dà_ottimo_intero {best_dual_bound}")
                    #primal_bound = best_dual_bound
                return #best_dual_bound, primal_bound, end_CG-start_CG
                #break
            
            if not new_columns_found: 
                assert sum(len(columns_to_add[n]) for n in range(N)) == 0
                if master.is_infeasible():
                    end_CG = perf_counter()
                    print("\nINFEASIBLE INSTANCE : we are still using a dummy column")
                    print(f"#iterations {iterations_count} CG_time {end_CG-start_CG:.3f}")
                    return #"INFEASIBLE INSTANCE", primal_bound, end_CG-start_CG
                else:
                    end_CG = perf_counter()
                    print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                    ### controllo se l'ottimo è già intero
                    if master.solution_is_integer(N):
                        print(f"CG_dà_ottimo_intero {best_dual_bound}")
                        #primal_bound = best_dual_bound
                    return #best_dual_bound, primal_bound, end_CG-start_CG
                    
                #break
            
            for n in range(N):
                if len(columns_to_add[n]) > 0:
                    assert len(columns_to_add[n]) == 1
                    master.add_column(n,columns_to_add[n][0],checked=False)

            #print("\n" + "*" * 80 + "\n")