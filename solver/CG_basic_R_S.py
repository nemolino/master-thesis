import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
import numpy as np
from random import shuffle

from exceptions import InfeasibleMaster, InfeasiblePricing
from column import Column
from CG_basic import Master, Pricing
from instance import Instance, InfeasibleOnBuilding
from model import ModelSolver
from math import ceil

options = {}
with open("../gurobi.lic") as f:
    lines = f.readlines()
    options["WLSACCESSID"] = lines[3].strip().split('=')[1]
    options["WLSSECRET"] = lines[4].strip().split('=')[1]
    options["LICENSEID"] = int(lines[5].strip().split('=')[1])
    options["OutputFlag"] = 0


class MasterExtended(Master):

    # extends Master with specific functionalities for R+S

    def most_selected_variable(self, still_to_fix):

        assert self.COLUMNS_POOL is not None
        assert self.z is not None

        COLUMNS_POOL = self.COLUMNS_POOL
        z = self.z

        N = len(still_to_fix)
        VALUE = [None] * N
        for n in range(N):
            VALUE[n] = dict()
            for u in still_to_fix[n]:
                VALUE[n][u] = dict()

        n_most = -1
        u_most = -1
        i_most = -1
        z_most = -1
        shuffled_N = list(range(N))
        shuffle(shuffled_N)
        for n in shuffled_N:
            for p in range(len(COLUMNS_POOL[n])):
                if z[n][p].X > 0: 
                    col = COLUMNS_POOL[n][p]
                    for u,i in enumerate(col.original_x):
                        if u in still_to_fix[n]:
                            if i not in VALUE[n][u]:
                                VALUE[n][u][i] = z[n][p].X
                            else:
                                VALUE[n][u][i] += z[n][p].X
                            if VALUE[n][u][i] > z_most:
                                
                                n_most = n 
                                u_most = u
                                i_most = i
                                z_most = VALUE[n][u][i]

        return n_most, u_most, i_most, z_most
    

class ColumnGeneration:

    
    def __init__(self, N=None, filenames=None):

        self.N = N                      # number of apps
        self.filenames = filenames


    def execute(self):

        assert self.N is not None and self.N > 0
        assert self.filenames is not None

        N = self.N
        network_filename, network_rp_filename, app_filenames, app_rp_filenames, apps_merged_filename, apps_merged_rp_filename = self.filenames

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
                    return
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
        
        # --- CG rounding + subMIPing --------------------------------------------------------------

        print("rounding_subMIPing")

        start_R = perf_counter()

        T_n = [APP[n]["T"] for n in range(N)]
        total_T = sum(T_n)
        fixed_variables_count = [0] * N

        complete_fixings_history = []

        still_to_fix = []
        for n in range(N):
            still_to_fix.append(set(range(T_n[n])))

        for n in range(N):
            for u in range(T_n[n]):
                if len(APP[n]["b_microservices_one"][u]) == 1:
                    still_to_fix[n].remove(u)
                    #not_still_to_fix[n].add(u)
                    complete_fixings_history.append((n,u,APP[n]["b_microservices_one"][u][0]))
                    fixed_variables_count[n] += 1


        while not sum(fixed_variables_count) >= 0.6 * total_T:

            n_most, u_most, i_most, z_most = master.most_selected_variable(still_to_fix)

            #print("n_most,u_most,i_most,z_most = ", n_most, u_most, i_most, z_most)

            ### rimuovo dal RMP le colonne in cui u_most non è in i_most
            to_remove = []
            removed_columns_count = 0
            for p in range(1,len(master.COLUMNS_POOL[n_most])):
                if master.COLUMNS_POOL[n_most][p].original_x[u_most] != i_most:
                    to_remove.append(p)
                    removed_columns_count += 1
            #print(f"removed_columns_count {removed_columns_count}")

            for p in to_remove:
                master.model.remove(master.z[n_most][p])
                master.z[n_most][p] = None
                master.COLUMNS_POOL[n_most][p] = None

            master.model.update()
            
            master.COLUMNS_POOL[n_most] = list(filter(lambda x: x is not None, master.COLUMNS_POOL[n_most]))
            master.z[n_most] = list(filter(lambda x: x is not None, master.z[n_most]))


            pp = pricings[n_most]
            x = pp.x
            pp.model.addConstr(x[u_most,i_most] == 1)
            pp.model.update()

            fixed_variables_count[n_most] += 1
            still_to_fix[n_most].remove(u_most)

            complete_fixings_history.append((n_most,u_most,i_most))

            if removed_columns_count == 0:
                continue

            ### CG

            iterations_count = 0
            best_dual_bound = 0

            while True:

                iterations_count += 1
                
                # potrebbe essere infeasible
                try:
                    λ_positive, μ_positive, η, master_obj_val = master.optimize()
                except InfeasibleMaster:
                    end_R = perf_counter()
                    print(f"\nINFEASIBLE MASTER, return {end_R-start_R:.3f}")
                    return  

                columns_to_add = []
                for n in range(N):
                    columns_to_add.append([])

                sum_reduced_costs = 0
                new_columns_found = False
                
                for n in range(N):

                    # potrebbe essere infeasible
                    try:
                        new_column, x, reduced_cost = pricings[n].optimize(λ_positive, μ_positive, η[n])
                    except InfeasiblePricing:
                        end_R = perf_counter()
                        print(f"\nINFEASIBLE PRICING, return {end_R-start_R:.3f}")
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
                
                
                dual_bound = master_obj_val + sum_reduced_costs
                    
                best_dual_bound = max(best_dual_bound, dual_bound)
                gap = master_obj_val-best_dual_bound
                perc_gap = 100 * gap / master_obj_val
                    
                if perc_gap <= STOP_GAP and not master.is_infeasible():   
                    #print(f"GAP # iterations : {iterations_count} master obj val : {master_obj_val}")
                    break
                
                if not new_columns_found: 
                    assert sum(len(columns_to_add[n]) for n in range(N)) == 0
                    if master.is_infeasible():
                        end_R = perf_counter()
                        print(f"\nINFEASIBLE INSTANCE : we are still using a dummy column, return {end_R-start_R:.3f}")
                        return
                    else:
                        pass
                        #print(f"# iterations : {iterations_count} master obj val : {master_obj_val}")
                    break
                
                for n in range(N):
                    if len(columns_to_add[n]) > 0:
                        assert len(columns_to_add[n]) == 1
                        master.add_column(n,columns_to_add[n][0],checked=True) # mettere checked a False


            ### controllo se l'ottimo è già intero

            if master.solution_is_integer(N):
                end_R = perf_counter()
                print(f"\nrounding_subMIPing SUCCESS perchè CG_ottimo_intero, primal bound {master.model.ObjVal}, return {end_R-start_R:.3f}")
                return

        
        instance = Instance.build(
            network_filename, 
            network_rp_filename, 
            app_filename=apps_merged_filename, 
            app_rp_filename=apps_merged_rp_filename
        )
        print("... solving restricted compact model")
        res = ModelSolver.optimize_model_restricted(instance, complete_fixings_history, T_n, start_R)
        end_R = perf_counter()
        print(f"{end_R-start_R:.3f} result {res}")
        return

