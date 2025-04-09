from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
import numpy as np
from copy import deepcopy

#from instance_unchecked import Instance, InfeasibleOnBuilding
from instance import Instance, InfeasibleOnBuilding
from heuristic_pricing_4 import HeuristicPricing
from model import ModelSolver
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
        #LOG.info(f"... master update in {end-start:.3f} s")

    
    # not in CG_basic : check it
    def add_column_possibly_duplicate(self, n, new_column):

        assert self.COLUMNS_POOL    is not None
        assert self.logger          is not None
        assert self.model           is not None
        assert self.z               is not None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        LOG = self.logger
        model = self.model
        z = self.z 
        
        if new_column in self.COLUMNS_POOL[n]:
            return False
    
        self.COLUMNS_POOL[n].append(new_column)

        # update master
        start = perf_counter()
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
        end = perf_counter()
        LOG.info(f"... master update in {end-start:.3f} s")

        return True
            

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


    # version of CG_basic
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
    
    # different wrt CG_basic
    def optimize_2(self, still_to_fix, fixed_columns):
        
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

        #LOG.info(f"... solving master   , {sum(len(COLUMNS_POOL[n]) for n in range(N))} columns available")
        #LOG.info("")

        for n in still_to_fix: 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.CONTINUOUS
                z[n][p].lb = 0.0

        model.optimize()

        if model.Status != GRB.OPTIMAL and model.Status != GRB.INFEASIBLE:
            raise RuntimeError("CG Master optimize : master status diverso da optimal e da infeasible")
        
        if model.Status == GRB.INFEASIBLE:
            raise InfeasibleMaster()

        # print(f"z* = {model.ObjVal} , master solve in {model.Runtime:.3f}\n")
        # LOG.info(f"z* = {model.ObjVal} , master solve in {model.Runtime:.3f}")
        # LOG.info("")

        # for n in range(N):
        #     print(f"application {n}, {len(COLUMNS_POOL[n])} patterns : {[round(z[n][p].X,2) for p in range(len(COLUMNS_POOL[n]))]}")
        #     LOG.debug(f"application {n}, {len(COLUMNS_POOL[n])} patterns : {[round(z[n][p].X,2) for p in range(len(COLUMNS_POOL[n]))]}")
        # print()
        # LOG.debug("")

        
        # start = perf_counter()

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
            if fixed_columns[n] is not None:
            #if n not in still_to_fix:
                η.append(None)
            else:
                value = model.getConstrByName(f"convexity_{n}").Pi
                assert value >= -epsilon
                η.append(value)
        η = tuple(η)

        # end = perf_counter()
        # # print(f"dual values retrieval in {end-start:.3f} s\n")
        # LOG.info(f"dual values retrieval in {end-start:.3f} s")
        # LOG.info("")

        
        # # controllo # vincoli
        # if len(model.getConstrs()) != len(I) * len(R) + len(A) * len(R) + N:
        #     raise RuntimeError(f"CG Master build_and_optimize : wrong number of constraints")

        # # controllo z* coi valori duali
        # z_calculation = 0
        # for i in I:
        #     for k in R:
        #         z_calculation += λ[(i,k)] * (-Q_nodes[(i,k)])
        # for (l,m) in A:
        #     for k in R:
        #         z_calculation += μ[((l,m),k)] * (-Q_links[((l,m),k)])
        # for n in range(N):
        #     z_calculation += η[n]
        # if abs(model.ObjVal - z_calculation) > 0.001:
        #     raise RuntimeError(f"CG Master build_and_optimize : mismatch between z* and its calculation from the duals")
        
        
        return λ_positive, μ_positive, η, model.ObjVal


    # not in CG_basic : check it
    def print_opt_sol(self):
    
        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data
        
        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        assert self.logger is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z
        LOG = self.logger

        print()
        #LOG.info("")
        for n in range(N): 
            print(f"application {n}")
            LOG.info(f"application {n}")
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0:
                    print(f"\tpattern {p:4d} : {z[n][p].X:.3f} {COLUMNS_POOL[n][p].original_x}")
                    LOG.info(f"\tpattern {p:4d} : {z[n][p].X:.3f}")


    """
    # not in CG_basic : check it
    def optimize_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        assert self.COLUMNS_POOL is not None
        assert self.logger is not None
        assert self.z is not None
        assert self.model is not None

        model_copy = deepcopy(self.model)
        z_copy = deepcopy(self.z)

        COLUMNS_POOL = self.COLUMNS_POOL
        #z = self.z
        # LOG = self.logger
        #model = self.model

        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z_copy[n][p].vtype = GRB.BINARY
        
        model_copy.optimize()
        
        assert model_copy.Status == GRB.OPTIMAL
        # LOG.info("")
        # LOG.info(f"discrete RMP : z* = {model.ObjVal}")
        print(f"\ndiscrete RMP : z* = {model_copy.ObjVal}")

    
    # not in CG_basic : check it
    def print_opt_sol_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data
        
        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        assert self.logger is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z
        LOG = self.logger

        patterns = []
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0.5: 
                    patterns.append(p)

        print(f"application patterns : {patterns}")
        LOG.info(f"application patterns : {patterns}")
    """

    
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


    # not in CG_basic
    def most_selected_column(self, pricings_to_solve):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        assert self.logger is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z

        n_most = -1
        p_most = -1
        z_most = -1
        for n in pricings_to_solve: 
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > z_most: 
                    z_most = z[n][p].X
                    n_most = n
                    p_most = p

        return n_most, p_most, z_most
    

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
    

    # not in CG_basic
    def solution_is_integer_2(self, still_to_fix):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None
        assert self.logger is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z

        for n in still_to_fix: 
            found = False
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X >= 0.999:
                    found = True
                    break
            if not found:
                return False

        return True
    
    
    # not in CG_basic : check it
    """
    def fix_pattern_to_1(self, n_most, p_most):

        assert self.model is not None
        assert self.z is not None
        

        model = self.model
        z = self.z

        model.addConstr(
            z[n_most][p_most] == 1, 
            name = f"fixed_pattern_for_app_{n_most}"
        )
        model.update()
    """
    

# questa classe è totalmente uguale a quella di CG_basic
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
    
    
class DiscreteMaster:


    def __init__(self, master):
        self.master = master


    def optimize_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data

        assert self.master.COLUMNS_POOL is not None
        assert self.master.logger is not None
        assert self.master.z is not None
        assert self.master.model is not None

        model = self.master.model
        z = self.master.z

        COLUMNS_POOL = self.master.COLUMNS_POOL
        #z = self.z
        # LOG = self.logger
        #model = self.model

        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.BINARY
        
        #model.setParam('TimeLimit', 10) #10*60)
        model.update()
        model.optimize()
        
        assert model.Status == GRB.Status.OPTIMAL #or model.Status == GRB.Status.TIME_LIMIT
        # LOG.info("")
        # LOG.info(f"discrete RMP : z* = {model.ObjVal}")
        #print(f"\ndiscrete RMP : z* = {model.ObjVal}")


    def print_opt_sol_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data
        
        assert self.master.COLUMNS_POOL is not None
        assert self.master.z is not None
        assert self.master.logger is not None

        COLUMNS_POOL = self.master.COLUMNS_POOL
        z = self.master.z
        #LOG = self.logger

        patterns = []
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0.5: 
                    patterns.append(p)

        print(f"application patterns : {patterns}")
        #LOG.info(f"application patterns : {patterns}")


    def restore_variables(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data

        assert self.master.COLUMNS_POOL is not None
        assert self.master.logger is not None
        assert self.master.z is not None
        assert self.master.model is not None

        model = self.master.model
        z = self.master.z
        COLUMNS_POOL = self.master.COLUMNS_POOL

        #model.setParam('TimeLimit', GRB.INFINITY)
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.CONTINUOUS
                z[n][p].lb = 0.0
        
        model.update()



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
            #LOG.info(f"{n}-th pricing build")
            
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
            #LOG.info(f"{n}-th pricing solved with dual values = 0, the column is the optimal mapping of {n}-th application")
            #LOG.info("")
        

        
        # print("*" * 80 + "\n")
        # LOG.info("*" * 80)
        # LOG.info("")

        """
        # --- CG ------------------------------------------------------------------------

        EPSILON = 0.001
        STOP_GAP = 0.1

        pricings_to_solve = list(range(N))

        dual_bound, primal_bound = None, None

        while len(pricings_to_solve) > 0:
            
            iterations_count = 0
            best_dual_bound = 0
            infeasible = False

            while True:

                iterations_count += 1
                print(f"iteration {iterations_count}\n")
                
                try:
                    λ_positive, μ_positive, η, master_obj_val = master.optimize()
                except InfeasibleMaster:
                    print('\033[91m' + f"\nINFEASIBLE MASTER" + '\033[0m')
                    infeasible = True
                    break

                
                # if iterations_count > 1:
                #     impr = 100 * (master_solutions[-1] - master.cur_optimal_sol) / master_solutions[-1]
                #     print(f"z* improvement : {impr:.5f} %\n") 
                    
                #master_solutions.append(master.cur_optimal_sol)

                #master.print_positive_dual_values_util(λ,μ,η)
                #print()

                #EXACT = iterations_count >= 2 and 100 * (prev_z_master - cur_z_master) / prev_z_master < 0.1 
                

                columns_to_add = []
                for n in range(N):
                    columns_to_add.append([])

                sum_reduced_costs = 0
                new_columns_found = False
                
                #for n in range(N):
                for n in pricings_to_solve:

                    # if n not in pricings_to_solve:
                    #     continue
                    
                    print('\033[91m' + f"... solving pricing of {n}-th application\t\t" + '\033[0m', end='')
                    LOG.info(f"... solving pricing of {n}-th application")

                    
                    # APP_n_T = APP[n]["T"]
                    # APP_n_D = APP[n]["D"]
                    # APP_n_q_microservices_R_core = APP[n]["q_microservices_R_core"]
                    # APP_n_q_connections_R_bandwidth = APP[n]["q_connections_R_bandwidth"]

                    # #try:

                    # heuristic_pricing_failed_at_this_iteration = True
                    # if (not EXACT) and heuristic_pricings_consecutive_fails[n] is not None:

                    #     # multiple pricing
                        
                    #     start = perf_counter()
                    #     results = heuristic_pricings[n].solve_multiple(
                    #         λ_positive, μ_positive, η[n] 
                    #     )
                    #     end = perf_counter()
                    #     print(f"app {n} - solving multiple heuristic pricing - time {end-start:.2f}")

                    #     if results is not None:
                    #         assert len(results) > 0
                    #         print(f"found {len(results)} negative reduced cost columns")
                    #         new_columns_found = True
                    #         for res in results:
                    #             assert res.cur_z < -EPSILON
                    #             heuristic_pricing_failed_at_this_iteration = False
                    #             heuristic_pricings_consecutive_fails[n] = 0
                    #             reduced_cost = res.cur_z

                    #             # building new_column
                    #             new_col_cost = sum(c[res.cur_x[u]] for u in range(APP_n_T))
                    #             new_col_q_core = np.zeros(I, dtype=np.int32)
                    #             for u in range(APP_n_T):
                    #                 i = res.cur_x[u]
                    #                 new_col_q_core[i] += APP_n_q_microservices_R_core[u]
                    #             new_col_q_bandwidth = np.zeros((I, I), dtype=np.int32)
                    #             for (u,v) in APP_n_D:
                    #                 i = res.cur_x[u]
                    #                 j = res.cur_x[v]
                    #                 for (l,m) in P[i][j]:
                    #                     new_col_q_bandwidth[l,m] += APP_n_q_connections_R_bandwidth[u,v]

                    #             new_column = Column(
                    #                 new_col_cost, new_col_q_core, new_col_q_bandwidth, res.cur_x
                    #             )
                    #             if new_column not in columns_to_add[n]:
                    #                 columns_to_add[n].append(new_column)
                    #     else:
                    #         print(f"found NO negative reduced cost columns")
                    #         heuristic_pricings_consecutive_fails[n] += 1
                    #         if heuristic_pricings_consecutive_fails[n] >= 3:
                    #             heuristic_pricings_consecutive_fails[n] = None


                    # if EXACT or heuristic_pricing_failed_at_this_iteration:

                    assert len(columns_to_add[n]) == 0

                    try:
                        new_column, x, reduced_cost = pricings[n].optimize(
                            λ_positive, μ_positive, η[n]
                        )
                    except InfeasiblePricing:
                        print('\033[91m' + f"\nINFEASIBLE PRICING" + '\033[0m')
                        infeasible = True
                        break
                
                    
                    # except RuntimeError as e:
                    #     print('\033[91m' + f" --> {e}" + '\033[0m', end='')
                    #     unfeasible_problem = True
                    #     break

                    # credo che vada spostato più giù

                    # ricalcolo costo ridotto
                    #if abs(reduced_cost - pricings[n].get_reduced_cost(x,λ,μ,η[n])) > epsilon:
                    #    raise RuntimeError(f"CG mismatch between c* (reduced costs) and its re-calculation")
                    
                    
                    if reduced_cost >= -EPSILON:
                        #columns_to_add[n].append(None)
                        print("--> NOT added to master")
                        LOG.info("--> NOT added to master")
                        continue
                    
                    sum_reduced_costs += reduced_cost
                    
                    print("--> added to master")
                    LOG.info("--> added to master")
                    new_columns_found = True
                    columns_to_add[n].append(new_column)
                    assert len(columns_to_add[n]) == 1
                
                if infeasible:
                    break
                    
                dual_bound = master_obj_val + sum_reduced_costs
                best_dual_bound = max(best_dual_bound, dual_bound)
                gap = master_obj_val-best_dual_bound
                perc_gap = 100 * gap / master_obj_val
                
                print(f"\n cur dual bound : {dual_bound:.5f}") 
                print(f"best dual bound : {best_dual_bound:.5f}")
                print(f"            gap : {gap:.5f} ( {perc_gap:.6f} % )")

                LOG.info("")
                # if new_columns_found:
                #     LOG.info(f" cur dual bound : {dual_bound:.5f}")
                LOG.info(f" cur dual bound : {dual_bound:.5f}")
                LOG.info(f"best dual bound : {best_dual_bound:.5f}")
                LOG.info(f"            gap : {gap:.5f} ( {perc_gap:.6f} % )")

                #dual_bounds.append(dual_bound)
                #best_dual_bounds.append(best_dual_bound)
                        
                if perc_gap <= STOP_GAP and not master.is_infeasible():
                    
                    print(f"\nSTOP, gap < {STOP_GAP} %")
                    LOG.info("")
                    LOG.info("*" * 80)
                    LOG.info("")
                    LOG.info(f"STOP, gap < {STOP_GAP} %")
                    break
                
                if not new_columns_found: 
                    assert sum(len(columns_to_add[n]) for n in range(N)) == 0
                    break
                
                LOG.info("")
                #for n in range(N):
                for n in pricings_to_solve:
                    if len(columns_to_add[n]) > 0:
                        assert len(columns_to_add[n]) == 1
                        #for col in columns_to_add[n]:
                        master.add_column(n,columns_to_add[n][0])

                print("\n" + "*" * 80 + "\n")
                LOG.info("")
                LOG.info("*" * 80)
                LOG.info("")

            end_CG = perf_counter()

            print(f"# iterations : {iterations_count}")
            print(f"master obj val : {master_obj_val}")
            print(f"LR dual bound : {best_dual_bound}")
            print(f"CG time : {end_CG-start_CG:.3f}s")

            LOG.info("")
            LOG.info(f"# iterations : {iterations_count}")
            LOG.info(f"master obj val : {master_obj_val}")
            LOG.info(f"LR dual bound : {best_dual_bound}")
            LOG.info(f"CG time : {end_CG-start_CG:.3f}s")

            master.print_opt_sol()
            
            # for n in range(N): 
            #     for u in range(APP[n]["T"]):
            #         print(f"used_nodes[{n}][{u}] = {used_nodes[n][u]}")
            # print()

            infeasible = master.is_infeasible()
            if infeasible:
                print('\033[91m' + f"\nINFEASIBLE INSTANCE : we are still using a dummy column" + '\033[0m')
                LOG.info("")
                LOG.warning("INFEASIBLE INSTANCE : we are still using a dummy column")
                return "INFEASIBLE INSTANCE", primal_bound, end_CG-start_CG
            
            if master.solution_is_integer(pricings_to_solve):
                print("integer solution")
                break
            
            dual_bound = best_dual_bound
            n_most, p_most, z_most = master.most_selected_column(pricings_to_solve)
            print(f"most selected pattern : app {n_most} pattern {p_most} with value {z_most}")
            
            # start_discrete_repair = perf_counter()
            # master.optimize_discrete()
            # master.print_opt_sol_discrete()
            # infeasible = master.is_infeasible()
            # if not infeasible:
            #     print(f"discrete RMP success")
            #     primal_bound = master.model.ObjVal
            # else:
            #     print(f"discrete RMP fail 1")
            # end_discrete_repair = perf_counter()
            # print(f"\ntotal discrete + repair time : {end_discrete_repair-start_discrete_repair:.2f} s")
            

            # fissa applicazione candidata e aggiorna tutto
            
            pricings_to_solve.remove(n_most)
            # update dei rhs dei pricings 
            for n in pricings_to_solve:
                pricings[n].update_rhs(master.COLUMNS_POOL[n_most][p_most])

            master.fix_pattern_to_1(n_most, p_most)
        

        if master.solution_is_integer(pricings_to_solve):
            print(f"master.solution_is_integer")
            λ_positive, μ_positive, η, master_obj_val = master.optimize()
            master.print_opt_sol()
            print(f"primal : {master.model.ObjVal}")
        else:
            print(f"infeasible = {infeasible}")

            start_discrete = perf_counter()
            master.optimize_discrete()
            master.print_opt_sol_discrete()
            infeasible = master.is_infeasible()
            if not infeasible:
                print(f"discrete RMP success")
                primal_bound = master.model.ObjVal
            else:
                print(f"discrete RMP fail 1")
            end_discrete = perf_counter()
            print(f"\ntotal discrete time : {end_discrete-start_discrete:.2f} s")
        """

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
                break
            
            if not new_columns_found: 
                print(f"\nno new columns found")
                assert sum(len(columns_to_add[n]) for n in range(N)) == 0
                if master.is_infeasible():
                    end_CG = perf_counter()
                    print("\nINFEASIBLE INSTANCE : we are still using a dummy column")
                    print(f"#iterations {iterations_count} CG_time {end_CG-start_CG:.3f}")
                    return #"INFEASIBLE INSTANCE", primal_bound, end_CG-start_CG
                else:
                    end_CG = perf_counter()
                    print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                break
            
            for n in range(N):
                if len(columns_to_add[n]) > 0:
                    assert len(columns_to_add[n]) == 1
                    master.add_column(n,columns_to_add[n][0],checked=False)

            #print("\n" + "*" * 80 + "\n")


        ### controllo se l'ottimo è già intero

        if master.solution_is_integer(N):
            print(f"CG_dà_ottimo_intero {best_dual_bound}")
            return
        

        # --- discrete RMP --------------------------------------------------------------

        obj_before = master.model.ObjVal
        #print(obj_before)

        dm = DiscreteMaster(master)

        dm.optimize_discrete()
        #dm.print_opt_sol_discrete()
        infeasible = dm.master.is_infeasible() #master.is_infeasible()
        if not infeasible:
            # bound time
            print(f"\ndiscrete_RMP_success {dm.master.model.ObjVal} {dm.master.model.Runtime}")
        else:
            # time
            print(f"\ndiscrete_RMP_fail {dm.master.model.Runtime}")
        dm.restore_variables()

        master.model.optimize()
        obj_after = master.model.ObjVal
        #print(obj_after)
        assert abs(obj_before - obj_after) < 0.001
        

        # --- CG pure diving ------------------------------------------------------------
        
        print("\nPureDiving\n")

        start_diving = perf_counter()
        
        still_to_fix = list(range(N))
        fixed_columns = [None] * N      # permetteranno di calcolare primal bound

        while len(still_to_fix) > 0: # maybe 

            ### trovo la colonna con valore frazionario più vicino a 1 e la fisso a 1 nel RMP
            n_most, p_most, z_most = master.most_selected_column(still_to_fix)
            print(f"most selected pattern : app {n_most} pattern {p_most} with value {z_most}")
            col_most = master.COLUMNS_POOL[n_most][p_most]
            fixed_columns[n_most] = col_most
            still_to_fix.remove(n_most)
            

            ### rimuovo dal RMP le variabili di colonna relative all'app appena fissata
            for p in range(len(master.COLUMNS_POOL[n_most])):
                master.model.remove(master.z[n_most][p])

            ### aggiorno i RHS del RMP 
            for i in range(I):
                if col_most.col_q_core[i] > 0:
                    master.model.getConstrByName(f"on_node_{i}_consumption_of_core").RHS += col_most.col_q_core[i]

            for (l,m) in A:
                if col_most.col_q_bandwidth[l,m] > 0:
                    master.model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").RHS += col_most.col_q_bandwidth[l,m]

            master.model.remove(
                master.model.getConstrByName(f"convexity_{n_most}")
            )

            master.model.update()

            master.COLUMNS_POOL[n_most] = []

            ### aggiorna RMP rimuovendo le colonne che lo renderebbero infeasible se messe a 1

            to_remove = set()

            for i in range(I):
                rhs = master.model.getConstrByName(f"on_node_{i}_consumption_of_core").RHS
                for n in still_to_fix:
                    for p in range(len(master.COLUMNS_POOL[n])):
                        if master.COLUMNS_POOL[n][p].col_q_core[i] > -rhs:
                            to_remove.add((n,p))
            
            # print(f"#col da rimuovere after consumption_of_core : {len(to_remove)}")

            for (l,m) in A:
                rhs = master.model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").RHS
                for n in still_to_fix:
                    for p in range(len(master.COLUMNS_POOL[n])):
                        if master.COLUMNS_POOL[n][p].col_q_bandwidth[l,m] > -rhs:
                            to_remove.add((n,p))
            
            # print(f"#col da rimuovere after consumption_of_bandwidth : {len(to_remove)}")
            # print(f"to_remove : {to_remove}")

            for (n,p) in to_remove:
                master.model.remove(master.z[n][p])
                master.z[n][p] = None
                master.COLUMNS_POOL[n][p] = None

            master.model.update()
            
            # removed_count = 0
            # removed_count_2 = 0

            for n in still_to_fix:
                # len_before = len(master.COLUMNS_POOL[n])
                master.COLUMNS_POOL[n] = list(filter(lambda x: x is not None, master.COLUMNS_POOL[n]))
                # len_after = len(master.COLUMNS_POOL[n])
                # removed_count += (len_before - len_after)

                # len_before = len(master.z[n])
                master.z[n] = list(filter(lambda x: x is not None, master.z[n]))
            #     len_after = len(master.z[n])
            #     removed_count_2 += (len_before - len_after)

            # assert len(to_remove) == removed_count == removed_count_2
            
            ### aggiorna i RHS dei pricing delle app non fissate

            for n in still_to_fix:

                pp = pricings[n]

                for i in range(I):
                    if col_most.col_q_core[i] > 0:
                        pp.consumption_core_constraint[i].RHS -= col_most.col_q_core[i]
                    
                for (l,m) in A:
                    if col_most.col_q_bandwidth[l,m] > 0:
                        pp.consumption_bandwidth_constraint[(l,m)].QCRHS -= col_most.col_q_bandwidth[l,m]

                pp.model.update()

            ### CG

            iterations_count = 0
            best_dual_bound = 0

            while True:

                iterations_count += 1
                
                #print(f"iteration diving {iterations_count}\n")
                
                # potrebbe essere infeasible
                try:
                    λ_positive, μ_positive, η, master_obj_val = master.optimize_2(still_to_fix, fixed_columns)
                except InfeasibleMaster:
                    end_diving = perf_counter()
                    print(f"\nINFEASIBLE MASTER, STOP {end_diving-start_diving:.3f}")
                    return #best_dual_bound, "... primal_bound ...", "... tempo ..." 

                columns_to_add = []
                for n in range(N):
                    columns_to_add.append([])

                #sum_reduced_costs = 0
                new_columns_found = False
                
                for n in still_to_fix:

                    #print('\033[91m' + f"... solving pricing of {n}-th application\t\t" + '\033[0m', end='')

                    # potrebbe essere infeasible
                    try:
                        new_column, x, reduced_cost = pricings[n].optimize(λ_positive, μ_positive, η[n])
                    except InfeasiblePricing:
                        end_diving = perf_counter()
                        print(f"\nINFEASIBLE PRICING, STOP {end_diving-start_diving:.3f}")
                        return #best_dual_bound, "... primal_bound ...", "... tempo ..." 
                    
                    if reduced_cost >= -EPSILON:
                        pass
                        #print("--> NOT added to master")
                    else:
                        #print("--> added to master")
                        #sum_reduced_costs += reduced_cost
                        new_columns_found = True
                        columns_to_add[n].append(new_column)
                        assert len(columns_to_add[n]) == 1
                
                
                # dual_bound = master_obj_val + sum_reduced_costs
                    
                # best_dual_bound = max(best_dual_bound, dual_bound)
                # gap = master_obj_val-best_dual_bound
                # perc_gap = 100 * gap / master_obj_val
                    
                # print(f"\n cur dual bound : {dual_bound:.5f}") 
                # print(f"best dual bound : {best_dual_bound:.5f}")
                # print(f"            gap : {gap:.5f} ( {perc_gap:.6f} % )")

                # if perc_gap <= STOP_GAP and not master.is_infeasible():   
                #     print(f"\nSTOP, gap < {STOP_GAP} %")
                #     end_CG = perf_counter()
                #     print(f"# iterations : {iterations_count}")
                #     print(f"master obj val : {master_obj_val}")
                #     print(f"LR dual bound : {best_dual_bound}")
                #     print(f"CG time : {end_CG-start_CG:.3f}s")
                #     break
                
                if not new_columns_found: 
                    print(f"\nno new columns found")
                    assert sum(len(columns_to_add[n]) for n in still_to_fix) == 0
                    if master.is_infeasible(still_to_fix):
                        end_diving = perf_counter()
                        print(f"\nINFEASIBLE still dummy, STOP {end_diving-start_diving:.3f}")
                        return #best_dual_bound, "... primal_bound ...", "... tempo ..." 
                    else:
                        print(f"#iter {iterations_count}")
                        #print(f"master obj val : {master_obj_val}")
                    break
                
                for n in still_to_fix:
                    if len(columns_to_add[n]) > 0:
                        assert len(columns_to_add[n]) == 1
                        master.add_column(n,columns_to_add[n][0],checked=False)

                #print("\n" + "*" * 80 + "\n")

            ### controllo se l'ottimo è già intero

            if master.solution_is_integer_2(still_to_fix):
                
                primal_bound = master.model.ObjVal
                for column in fixed_columns:
                    if column is not None:
                        primal_bound += column.col_cost
                end_diving = perf_counter()
                print("CG_dà_già_ottimo_intero")
                print(f"DIVING_SUCCESS primal_bound {primal_bound} {end_diving-start_diving:.3f}")
                return #best_dual_bound, primal_bound, end_CG-start_CG

        print("COSA FACCIO QUI?")


        # --- CG subMIPing --------------------------------------------------------------
        """
        start_subMIPing = perf_counter()
        
        constraints_to_add = []

        for n in range(N):
            
            used_nodes = []
            for _ in range(APP[n]["T"]):
                used_nodes.append(set())

            for p in range(1,len(master.COLUMNS_POOL[n])):
                #if master.z[n][p].X > 0.001: 
                for u in range(APP[n]["T"]):
                    used_nodes[u].add(master.COLUMNS_POOL[n][p].original_x[u])

            constraints_to_add.append(used_nodes)

        assert len(constraints_to_add) == N
        for n in range(N):
            assert len(constraints_to_add[n]) == APP[n]["T"]


        for n in range(N):
            for u in range(APP[n]["T"]):
                print(n,u, constraints_to_add[n][u])

        instance = Instance.build(
            network_filename, 
            network_rp_filename, 
            app_filename=apps_merged_filename, 
            app_rp_filename=apps_merged_rp_filename
        )

        res = ModelSolver.optimize_model_restricted(instance, constraints_to_add)
        end_subMIPing = perf_counter()

        print(f"subMIPing result : {res}")
        print(f"subMIPing time : {end_subMIPing-start_subMIPing:.2f} s\n\n")
        """

        # --- CG diving with subMIPing --------------------------------------------------
        """
        print("\n--- DIVING with subMIPing -------\n")
        start_diving_subMIPing = perf_counter()
        still_to_fix = list(range(N))
        fixed_columns = [None] * N      # permetteranno di calcolare primal bound
        
        fixed_columns_in_order = []
        infeasible_master = False
        
        k = N // 2

        while len(still_to_fix) > k: # maybe 

            ### trovo la colonna con valore frazionario più vicino a 1 e la fisso a 1 nel RMP
            n_most, p_most, z_most = master.most_selected_column(still_to_fix)
            print(f"most selected pattern : app {n_most} pattern {p_most} with value {z_most}")
            col_most = master.COLUMNS_POOL[n_most][p_most]
            fixed_columns[n_most] = col_most
            still_to_fix.remove(n_most)
            fixed_columns_in_order.append(n_most)

            if len(still_to_fix) == k:
                break
            

            ### rimuovo dal RMP le variabili di colonna relative all'app appena fissata
            for p in range(len(master.COLUMNS_POOL[n_most])):
                master.model.remove(master.z[n_most][p])

            ### aggiorno i RHS del RMP 
            for i in range(I):
                if col_most.col_q_core[i] > 0:
                    master.model.getConstrByName(f"on_node_{i}_consumption_of_core").RHS += col_most.col_q_core[i]

            for (l,m) in A:
                if col_most.col_q_bandwidth[l,m] > 0:
                    master.model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").RHS += col_most.col_q_bandwidth[l,m]

            master.model.remove(
                master.model.getConstrByName(f"convexity_{n_most}")
            )

            master.model.update()

            master.COLUMNS_POOL[n_most] = []

            ### aggiorna RMP rimuovendo le colonne che lo renderebbero infeasible se messe a 1

            to_remove = set()

            for i in range(I):
                rhs = master.model.getConstrByName(f"on_node_{i}_consumption_of_core").RHS
                for n in still_to_fix:
                    for p in range(len(master.COLUMNS_POOL[n])):
                        if master.COLUMNS_POOL[n][p].col_q_core[i] > -rhs:
                            to_remove.add((n,p))
            
            print(f"#col da rimuovere after consumption_of_core : {len(to_remove)}")

            for (l,m) in A:
                rhs = master.model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").RHS
                for n in still_to_fix:
                    for p in range(len(master.COLUMNS_POOL[n])):
                        if master.COLUMNS_POOL[n][p].col_q_bandwidth[l,m] > -rhs:
                            to_remove.add((n,p))
            
            print(f"#col da rimuovere after consumption_of_bandwidth : {len(to_remove)}")
            print(f"to_remove : {to_remove}")

            for (n,p) in to_remove:
                master.model.remove(master.z[n][p])
                master.z[n][p] = None
                master.COLUMNS_POOL[n][p] = None

            master.model.update()
            
            removed_count = 0
            removed_count_2 = 0

            for n in still_to_fix:
                len_before = len(master.COLUMNS_POOL[n])
                master.COLUMNS_POOL[n] = list(filter(lambda x: x is not None, master.COLUMNS_POOL[n]))
                len_after = len(master.COLUMNS_POOL[n])
                removed_count += (len_before - len_after)

                len_before = len(master.z[n])
                master.z[n] = list(filter(lambda x: x is not None, master.z[n]))
                len_after = len(master.z[n])
                removed_count_2 += (len_before - len_after)

            assert len(to_remove) == removed_count == removed_count_2
            
            ### aggiorna i RHS dei pricing delle app non fissate

            for n in still_to_fix:

                pp = pricings[n]

                for i in range(I):
                    if col_most.col_q_core[i] > 0:
                        pp.consumption_core_constraint[i].RHS -= col_most.col_q_core[i]
                    
                for (l,m) in A:
                    if col_most.col_q_bandwidth[l,m] > 0:
                        pp.consumption_bandwidth_constraint[(l,m)].QCRHS -= col_most.col_q_bandwidth[l,m]

                pp.model.update()

            ### CG

            iterations_count = 0
            best_dual_bound = 0

            while True:

                iterations_count += 1
                
                print(f"iteration diving {iterations_count}\n")
                
                # potrebbe essere infeasible
                try:
                    λ_positive, μ_positive, η, master_obj_val = master.optimize(still_to_fix)
                except InfeasibleMaster:
                    print('\033[91m' + f"\nINFEASIBLE MASTER" + '\033[0m')
                    infeasible_master = False
                    break
                    print("STOP")
                    return best_dual_bound, "... primal_bound ...", "... tempo ..." 

                columns_to_add = []
                for n in range(N):
                    columns_to_add.append([])

                #sum_reduced_costs = 0
                new_columns_found = False
                
                for n in still_to_fix:

                    print('\033[91m' + f"... solving pricing of {n}-th application\t\t" + '\033[0m', end='')

                    # potrebbe essere infeasible
                    try:
                        new_column, x, reduced_cost = pricings[n].optimize(λ_positive, μ_positive, η[n])
                    except InfeasiblePricing:
                        print('\033[91m' + f"\nINFEASIBLE PRICING" + '\033[0m')
                        print("STOP")
                        return best_dual_bound, "... primal_bound ...", "... tempo ..." 
                    
                    if reduced_cost >= -EPSILON:
                        print("--> NOT added to master")
                    else:
                        print("--> added to master")
                        #sum_reduced_costs += reduced_cost
                        new_columns_found = True
                        columns_to_add[n].append(new_column)
                        assert len(columns_to_add[n]) == 1
                
                
                # dual_bound = master_obj_val + sum_reduced_costs
                    
                # best_dual_bound = max(best_dual_bound, dual_bound)
                # gap = master_obj_val-best_dual_bound
                # perc_gap = 100 * gap / master_obj_val
                    
                # print(f"\n cur dual bound : {dual_bound:.5f}") 
                # print(f"best dual bound : {best_dual_bound:.5f}")
                # print(f"            gap : {gap:.5f} ( {perc_gap:.6f} % )")

                # if perc_gap <= STOP_GAP and not master.is_infeasible():   
                #     print(f"\nSTOP, gap < {STOP_GAP} %")
                #     end_CG = perf_counter()
                #     print(f"# iterations : {iterations_count}")
                #     print(f"master obj val : {master_obj_val}")
                #     print(f"LR dual bound : {best_dual_bound}")
                #     print(f"CG time : {end_CG-start_CG:.3f}s")
                #     break
                
                if not new_columns_found: 
                    print(f"\nno new columns found")
                    assert sum(len(columns_to_add[n]) for n in still_to_fix) == 0
                    if master.is_infeasible(still_to_fix):
                        print('\033[91m' + f"\nINFEASIBLE INSTANCE : we are still using a dummy column" + '\033[0m')
                        print("STOP")
                        return best_dual_bound, "... primal_bound ...", "... tempo ..." 
                    else:
                        print(f"# iterations : {iterations_count}")
                        print(f"master obj val : {master_obj_val}")
                    break
                
                for n in still_to_fix:
                    if len(columns_to_add[n]) > 0:
                        assert len(columns_to_add[n]) == 1
                        master.add_column(n,columns_to_add[n][0],checked=False)

                print("\n" + "*" * 80 + "\n")

            ### controllo se l'ottimo è già intero

            if master.solution_is_integer_2(still_to_fix):
                
                primal_bound = master.model.ObjVal
                for column in fixed_columns:
                    if column is not None:
                        primal_bound += column.col_cost
                print("CG dà già l'ottimo intero")
                print(f"DIVING SUCCESS con primal bound {primal_bound}")
                return best_dual_bound, primal_bound, end_CG-start_CG


        #print("COSA FACCIO QUI?")

        if infeasible_master:
            print(infeasible_master, still_to_fix, fixed_columns, fixed_columns_in_order)
            # tolgo gli ultimi due fissaggi
            fixed_columns[fixed_columns[-1]] = None 
            fixed_columns[fixed_columns[-2]] = None 
            still_to_fix.append(fixed_columns[-1])
            still_to_fix.append(fixed_columns[-2])
            fixed_columns = fixed_columns[:-2]
        else:
            print(infeasible_master, still_to_fix, fixed_columns, fixed_columns_in_order)
            assert len(still_to_fix) == 5

        for c in fixed_columns_in_order:    assert fixed_columns[c] is not None
        for c in still_to_fix:              assert fixed_columns[c] is None

        instance = Instance.build(
            network_filename, 
            network_rp_filename, 
            app_filename=apps_merged_filename, 
            app_rp_filename=apps_merged_rp_filename
        )

        app_sizes = [APP[n]["T"] for n in range(N)]
        res = ModelSolver.optimize_model_restricted_2(instance, fixed_columns, app_sizes)
        end_diving_subMIPing = perf_counter()

        print(f"diving with subMIPing result : {res}")
        print(f"diving with subMIPing time : {end_diving_subMIPing-start_diving_subMIPing:.2f} s\n\n")
        """

        return #best_dual_bound, primal_bound, end_CG-start_CG
    

# cg diving
# https://www.researchgate.net/publication/286907487_Primal_Heuristics_for_Branch_and_Price_The_Assets_of_Diving_Methods
# https://www.gerad.ca/colloques/ColumnGeneration2016/PDF/Sadykov.pdf
# https://homes.di.unimi.it/cordone/courses/2024-mhco/Lez06.pdf

