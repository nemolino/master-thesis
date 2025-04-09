import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
import numpy as np

from exceptions import InfeasibleMaster, InfeasiblePricing
from column import Column
from CG_basic import Master, Pricing
from instance import Instance, InfeasibleOnBuilding
from math import ceil

options = {}
with open("../gurobi.lic") as f:
    lines = f.readlines()
    options["WLSACCESSID"] = lines[3].strip().split('=')[1]
    options["WLSSECRET"] = lines[4].strip().split('=')[1]
    options["LICENSEID"] = int(lines[5].strip().split('=')[1])
    options["OutputFlag"] = 0


class MasterExtended(Master):

    # extends Master with specific functionalities for DLRMP and PD

    def optimize_2(self, still_to_fix, fixed_columns):
        
        assert self.COLUMNS_POOL    is not None
        assert self.z               is not None
        assert self.model           is not None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z
        model = self.model

        for n in still_to_fix: 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.CONTINUOUS
                z[n][p].lb = 0.0

        model.optimize()

        if model.Status != GRB.OPTIMAL and model.Status != GRB.INFEASIBLE:
            raise RuntimeError("CG Master optimize : master status diverso da optimal e da infeasible")
        
        if model.Status == GRB.INFEASIBLE:
            raise InfeasibleMaster()

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
                η.append(None)
            else:
                value = model.getConstrByName(f"convexity_{n}").Pi
                assert value >= -epsilon
                η.append(value)
        η = tuple(η)
        
        return λ_positive, μ_positive, η, model.ObjVal


    def most_selected_column(self, pricings_to_solve):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None

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
    
    
    def solution_is_integer_2(self, still_to_fix):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None

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
    

class DiscreteMaster:


    def __init__(self, master):
        self.master = master


    def optimize_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data

        assert self.master.COLUMNS_POOL is not None
        assert self.master.z is not None
        assert self.master.model is not None

        model = self.master.model
        z = self.master.z

        COLUMNS_POOL = self.master.COLUMNS_POOL

        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.BINARY
        
        model.update()
        model.optimize()
        
        assert model.Status == GRB.Status.OPTIMAL


    def restore_variables(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data

        assert self.master.COLUMNS_POOL is not None
        assert self.master.z is not None
        assert self.master.model is not None

        model = self.master.model
        z = self.master.z
        COLUMNS_POOL = self.master.COLUMNS_POOL

        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                z[n][p].vtype = GRB.CONTINUOUS
                z[n][p].lb = 0.0
        
        model.update()


    """
    def print_opt_sol_discrete(self):

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.master.data
        
        assert self.master.COLUMNS_POOL is not None
        assert self.master.z is not None

        COLUMNS_POOL = self.master.COLUMNS_POOL
        z = self.master.z

        patterns = []
        for n in range(N): 
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0.5: 
                    patterns.append(p)

        print(f"application patterns : {patterns}")
    """


class ColumnGeneration:

    
    def __init__(self, N=None, filenames=None, logger=None):

        self.N = N                      # number of apps
        self.filenames = filenames
        self.logger = logger


    def execute(self):

        assert self.N is not None and self.N > 0
        assert self.filenames is not None

        N = self.N
        network_filename, network_rp_filename, app_filenames, app_rp_filenames = self.filenames

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
                return
                
            APP[n]["b_microservices_zero"] = b_microservices_zero
            APP[n]["b_microservices_one"] = b_microservices_one
            APP[n]["b_connections_zero_not_implied"] = b_connections_zero_not_implied
            APP[n]["b_connections_one"] = b_connections_one
            APP[n]["b_connections_one_actual"] = b_connections_one_actual

        # --- Building CG structures ----------------------------------------------------

        dummy_columns_costs = [APP[n]["T"] * max_node_cost for n in range(N)]
        number_of_microservices = [APP[n]["T"] for n in range(N)]

        master = MasterExtended(
            (I,A,N,Q_nodes_R_core,Q_links_R_bandwidth), 
            dummy_columns_costs, number_of_microservices
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
                    )
                )
            )
            
            # adding column of the optimal mapping of n-th application
            try:
                first_column, _, _ = pricings[n].optimize(
                    λ_positive=(), μ_positive=(), η_n=0
                )
            except InfeasiblePricing:
                print(f"INFEASIBLE INSTANCE : {n}-th pricing infeasible with dual values = 0, meaning that you cannot map {n}-th application")
                return

            master.add_column(n, first_column)

        # --- CG ------------------------------------------------------------------------

        EPSILON = 0.001
        STOP_GAP = 0.1

        iterations_count = 0
        best_dual_bound = 0

        while True:

            iterations_count += 1
            
            # non deve essere infeasible perchè ho le dummy
            λ_positive, μ_positive, η, master_obj_val = master.optimize() 

            columns_to_add = []
            for n in range(N):
                columns_to_add.append([])

            sum_reduced_costs = 0
            new_columns_found = False
            
            for n in range(N):

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
                else:
                    end_CG = perf_counter()
                    print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                break
            
            for n in range(N):
                if len(columns_to_add[n]) > 0:
                    assert len(columns_to_add[n]) == 1
                    master.add_column(n,columns_to_add[n][0],checked=False)


        ### controllo se l'ottimo è già intero

        if master.solution_is_integer(N):
            print(f"CG_dà_ottimo_intero {best_dual_bound}")
            return
        

        # --- discrete LRMP --------------------------------------------------------------
        obj_before = master.model.ObjVal

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
        
        print("\nPureDiving ", end='')

        start_diving = perf_counter()
        
        still_to_fix = list(range(N))
        fixed_columns = [None] * N      # permetteranno di calcolare primal bound

        partial_cost = 0

        while len(still_to_fix) > 0: 

            ### trovo la colonna con valore frazionario più vicino a 1 e la fisso a 1 nel RMP
            n_most, p_most, z_most = master.most_selected_column(still_to_fix)
            #print(f"most selected pattern : app {n_most} pattern {p_most} with value {z_most}")
            col_most = master.COLUMNS_POOL[n_most][p_most]
            fixed_columns[n_most] = col_most
            still_to_fix.remove(n_most)
            partial_cost += col_most.col_cost
            

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
            
            for (l,m) in A:
                rhs = master.model.getConstrByName(f"on_link_({l},{m})_consumption_of_bandwidth").RHS
                for n in still_to_fix:
                    for p in range(len(master.COLUMNS_POOL[n])):
                        if master.COLUMNS_POOL[n][p].col_q_bandwidth[l,m] > -rhs:
                            to_remove.add((n,p))
            
            for (n,p) in to_remove:
                master.model.remove(master.z[n][p])
                master.z[n][p] = None
                master.COLUMNS_POOL[n][p] = None

            master.model.update()
            
            for n in still_to_fix:
                master.COLUMNS_POOL[n] = list(filter(lambda x: x is not None, master.COLUMNS_POOL[n]))
                master.z[n] = list(filter(lambda x: x is not None, master.z[n]))
            
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
                
                # potrebbe essere infeasible
                try:
                    λ_positive, μ_positive, η, master_obj_val = master.optimize_2(still_to_fix, fixed_columns)
                except InfeasibleMaster:
                    end_diving = perf_counter()
                    print(f"INFEASIBLE MASTER, STOP {end_diving-start_diving:.3f}")
                    return

                columns_to_add = []
                for n in range(N):
                    columns_to_add.append([])

                sum_reduced_costs = 0
                new_columns_found = False
                
                for n in still_to_fix:

                    # potrebbe essere infeasible
                    try:
                        new_column, x, reduced_cost = pricings[n].optimize(λ_positive, μ_positive, η[n])
                    except InfeasiblePricing:
                        end_diving = perf_counter()
                        print(f"INFEASIBLE PRICING, STOP {end_diving-start_diving:.3f}")
                        return
                    
                    if reduced_cost >= -EPSILON:
                        pass
                        #print("--> NOT added to master")
                    else:
                        #print("--> added to master")
                        sum_reduced_costs += reduced_cost
                        new_columns_found = True
                        columns_to_add[n].append(new_column)
                        assert len(columns_to_add[n]) == 1
                
                
                dual_bound = master_obj_val + sum_reduced_costs + partial_cost
                    
                best_dual_bound = max(best_dual_bound, dual_bound)
                gap = master_obj_val-best_dual_bound
                perc_gap = 100 * gap / master_obj_val

                if perc_gap <= STOP_GAP and not master.is_infeasible(still_to_fix):   
                    #print(f"\nSTOP, gap < {STOP_GAP} % , #iter  {iterations_count}")
                    break
                
                if not new_columns_found: 
                    #print(f"\nno new columns found")
                    assert sum(len(columns_to_add[n]) for n in still_to_fix) == 0
                    if master.is_infeasible(still_to_fix):
                        end_diving = perf_counter()
                        print(f"INFEASIBLE still dummy, STOP {end_diving-start_diving:.3f}")
                        return
                    else:
                        pass
                        #print(f"#iter {iterations_count}")
                    break
                
                for n in still_to_fix:
                    if len(columns_to_add[n]) > 0:
                        assert len(columns_to_add[n]) == 1
                        master.add_column(n,columns_to_add[n][0],checked=False)
            ### controllo se l'ottimo è già intero

            if master.solution_is_integer_2(still_to_fix):
                
                primal_bound = partial_cost + master.model.ObjVal
                end_diving = perf_counter()
                #print("CG_dà_già_ottimo_intero")
                print(f"PURE_DIVING_SUCCESS primal_bound {primal_bound} {end_diving-start_diving:.3f}")
                return

        return
