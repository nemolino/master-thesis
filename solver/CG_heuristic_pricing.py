from time import perf_counter
import numpy as np
from math import ceil
import gurobipy as gp
from gurobipy import GRB

from exceptions import InfeasibleMaster, InfeasiblePricing
from column import Column
from CG_basic import Master, Pricing
from instance import Instance, InfeasibleOnBuilding
from heuristic_pricing import HeuristicPricing


options = {}
with open("../gurobi.lic") as f:
    lines = f.readlines()
    options["WLSACCESSID"] = lines[3].strip().split('=')[1]
    options["WLSSECRET"] = lines[4].strip().split('=')[1]
    options["LICENSEID"] = int(lines[5].strip().split('=')[1])
    options["OutputFlag"] = 0


class MasterHP(Master):


    def add_column_possibly_duplicate(self, n, new_column):

        assert self.COLUMNS_POOL    is not None
        assert self.model           is not None
        assert self.z               is not None

        (
            I,A,N,
            Q_nodes_R_core,Q_links_R_bandwidth
        ) = self.data

        model = self.model
        z = self.z 
        
        if new_column in self.COLUMNS_POOL[n]:
            return False
    
        self.COLUMNS_POOL[n].append(new_column)

        # update master
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

        return True
    
    
class ColumnGeneration:

    
    def __init__(self, N=None, filenames=None):

        self.N = N                      # number of apps
        self.filenames = filenames


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

        master = MasterHP(
            (I,A,N,Q_nodes_R_core,Q_links_R_bandwidth), 
            dummy_columns_costs, number_of_microservices
        )

        pricings = []
        heuristic_pricings = []

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

            heuristic_pricings.append(
                HeuristicPricing(
                    data = (
                        I,c,P,a2,
                        APP[n]["T"], APP[n]["D"],
                        Q_nodes_R_core, Q_links_R_bandwidth,
                        APP[n]["q_microservices_R_core"], 
                        APP[n]["q_connections_R_bandwidth"],
                        APP[n]["b_microservices_one"],
                        APP[n]["b_connections_one"]
                    )
                )
            )
            
        """
        for n in range(N):

            for μ in (0.2, 0.4, 0.6, 0.8, 1.0):

                feasible_columns = heuristic_pricings[n].solve_multiple(
                    [], [], 0, μ, get_feasible_col=True
                )
                
                if feasible_columns is not None:
                    
                    columns_to_add = []

                    for new_col_cost, new_col_q_core, new_col_q_bandwidth, col_x in feasible_columns:

                        new_column = Column(
                            new_col_cost, new_col_q_core, new_col_q_bandwidth, col_x
                        )
                        if new_column not in columns_to_add:
                            columns_to_add.append(new_column)
                    
                    for col in columns_to_add:
                        master.add_column(n,col,checked=False)
        """

        # --- CG with HP ------------------------------------------------------------------------

        EPSILON = 0.001
        STOP_GAP = 0.1
        iterations_count = 0
        best_dual_bound = 0

        prev_z_master, cur_z_master = None, None
        heuristic_pricings_consecutive_fails = [0] * N

        while True:

            iterations_count += 1
            
            # non deve essere infeasible perchè ho le dummy
            λ_positive, μ_positive, η, master_obj_val = master.optimize()
            prev_z_master, cur_z_master = cur_z_master, master_obj_val
            
            EXACT = iterations_count >= 2 and 100 * (prev_z_master - cur_z_master) / prev_z_master < 5

            columns_to_add = []
            for n in range(N):
                columns_to_add.append([])

            sum_reduced_costs = 0
            new_columns_found = False
            
            for n in range(N):
                
                heuristic_pricing_failed_at_this_iteration = True
                if (not EXACT) and heuristic_pricings_consecutive_fails[n] is not None:

                    # multiple pricing
                    
                    results = heuristic_pricings[n].solve_multiple(
                        λ_positive, μ_positive, η[n] 
                    )
                    
                    if results is not None:
                        assert len(results) > 0
                        new_columns_found = True
                        heuristic_pricing_failed_at_this_iteration = False
                        heuristic_pricings_consecutive_fails[n] = 0

                        for new_col_cost, new_col_q_core, new_col_q_bandwidth, col_x in results:

                            new_column = Column(
                                new_col_cost, new_col_q_core, new_col_q_bandwidth, col_x
                            )
                            if new_column not in columns_to_add[n]:
                                columns_to_add[n].append(new_column)
                        
                    else:
                        heuristic_pricings_consecutive_fails[n] = None
                        

                if EXACT or heuristic_pricing_failed_at_this_iteration:

                    assert len(columns_to_add[n]) == 0

                    new_column, x, reduced_cost = pricings[n].optimize(
                        λ_positive, μ_positive, η[n]
                    )
                    
                    if reduced_cost >= -EPSILON:
                        pass
                        #print("--> NOT added to master")
                    else:
                        sum_reduced_costs += reduced_cost
                        #print("--> added to master")
                        new_columns_found = True
                        columns_to_add[n].append(new_column)
                        assert len(columns_to_add[n]) == 1
                
            if EXACT or not new_columns_found:  

                dual_bound = master_obj_val + sum_reduced_costs

                best_dual_bound = max(best_dual_bound, dual_bound)
                gap = master_obj_val-best_dual_bound
                perc_gap = 100 * gap / master_obj_val
                        
                if perc_gap <= STOP_GAP and not master.is_infeasible():
                    
                    end_CG = perf_counter()

                    # stop solo se ho smesso di usare dummy 
                    print(f"\nSTOP, gap < {STOP_GAP} %")
                    print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                    if master.solution_is_integer(N):
                        print(f"CG_dà_ottimo_intero {best_dual_bound}")
                    return
            
            if not new_columns_found: 
                assert sum(len(columns_to_add[n]) for n in range(N)) == 0
                if master.is_infeasible():
                    end_CG = perf_counter()
                    print("\nINFEASIBLE INSTANCE : we are still using a dummy column")
                    print(f"#iterations {iterations_count} CG_time {end_CG-start_CG:.3f}")
                    return
                else:
                    end_CG = perf_counter()
                    print(f"#iterations {iterations_count} LR_dual_bound {best_dual_bound} CG_time {end_CG-start_CG:.3f}")
                    if master.solution_is_integer(N):
                        print(f"CG_dà_ottimo_intero {best_dual_bound}")
                    return
                
            
            for n in range(N):
                if len(columns_to_add[n]) > 0:
                    for col in columns_to_add[n]:
                        master.add_column(n,col,checked=False)
