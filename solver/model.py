import gurobipy as gp
from gurobipy import GRB
from time import perf_counter

OPTIONS = {}
with open("../gurobi.lic") as f:
    lines = f.readlines()
    OPTIONS["WLSACCESSID"] = lines[3].strip().split('=')[1]
    OPTIONS["WLSSECRET"] = lines[4].strip().split('=')[1]
    OPTIONS["LICENSEID"] = int(lines[5].strip().split('=')[1])
    OPTIONS["OutputFlag"] = 0


class ModelSolver():


    @staticmethod
    def build_model(ins, vtype=None):

        (   
            I, A, c, P, T, D, 
            b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual,
            Q_nodes_R_core, q_microservices_R_core,
            Q_links_R_bandwidth, q_connections_R_bandwidth
        ) = (
            ins.I, ins.A, ins.c, ins.P, ins.T, ins.D,
            ins.b_microservices_zero, ins.b_connections_zero_not_implied, ins.b_connections_one_actual,
            ins.Q_nodes_R_core, ins.q_microservices_R_core,
            ins.Q_links_R_bandwidth, ins.q_connections_R_bandwidth
        )

        """
        if verbose:
            OPTIONS["OutputFlag"] = 1
        """

        model = gp.Model("CAVIA", env=gp.Env(params=OPTIONS))

        """
        OPTIONS["OutputFlag"] = 0
        """

        if vtype is None or vtype == GRB.BINARY:
            x = model.addVars(T, I, vtype=GRB.BINARY, name = "x")
        elif vtype == GRB.CONTINUOUS:
            x = model.addVars(T, I, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name = "x")
        else:
            raise ValueError("invalid vtype")

        model.setObjective(
            gp.quicksum(c[i] * x[u,i] for u in range(T) for i in range(I)), 
            GRB.MINIMIZE
        )

        model.addConstrs(
            (
                gp.quicksum(x[u,i] for i in range(I)) == 1 
                for u in range(T)
            ), 
            name='microservice_u_mapped_to_exactly_one_node'
        )

        model.addConstrs(
            (
                x[u,i] == 0 
                for u in range(T) 
                    for i in b_microservices_zero[u]
            ), 
            name='microservice_u_not_mappable_to_node_i'
        )
        
        model.addConstrs(
            (
                x[u,i] + x[v,j] <= 1
                for (u,v) in D
                    for (i,j) in b_connections_zero_not_implied[u][v]
                    # for (i,j) in b_connections_zero[u][v] 
            ),
            name='microservices_(u,v)_not_mappable_to_nodes_(i,j)'
        )

        model.addConstrs(
            (
                gp.quicksum(q_microservices_R_core[u] * x[u,i] for u in range(T)) <= Q_nodes_R_core[i]
                for i in range(I)
            ),
            name='on_node_i_consumption_of_resource_core'
        )

        lhs = dict()
        for (l,m) in A:
            lhs[(l,m)] = gp.QuadExpr(0)
            
        for (u,v) in D:
            for (i,j) in b_connections_one_actual[u][v]:
                for (l,m) in P[i][j]:
                    lhs[(l,m)].add(q_connections_R_bandwidth[u,v] * x[u,i] * x[v,j])

        model.addConstrs(
            (
                lhs[(l,m)] <= Q_links_R_bandwidth[l,m]
                for (l,m) in A
            ),
            name='on_link_(l,m)_consumption_of_resource_bandwidth'
        )
        
        return model, x
    

    @staticmethod
    def optimize_model_continuous_relaxation(instance):

        start = perf_counter()
        model, x = ModelSolver.build_model(instance, vtype=GRB.CONTINUOUS)
        end = perf_counter()
        print(f"{end-start:.3f} s is the total build time of the model")
        
        model.setParam('NodeLimit', GRB.INFINITY)
        model.setParam('TimeLimit', 1*60)
        model.update()
        model.optimize()

        if model.Status == GRB.Status.OPTIMAL:
            print(f"solve time : {model.Runtime}, OPTIMAL, # explored nodes : {model.NodeCount}, z* = {model.ObjVal}")
        elif model.Status == GRB.Status.INFEASIBLE:
            print(f"solve time : {model.Runtime}, INFEASIBLE, # explored nodes : {model.NodeCount}")
        elif model.Status == GRB.Status.TIME_LIMIT:
            print(f"solve time : {model.Runtime}, TIME_LIMIT, # explored nodes : {model.NodeCount}, z* = {model.ObjVal}")
        else:
            raise RuntimeError(f"optimize_model_continuous_relaxation : unexpected status : {model.Status}")
        
        for var in model.getVars():
            var.setAttr('vtype', GRB.BINARY)
        model.update()
        

    @staticmethod
    def optimize_model(instance):

        model, x = ModelSolver.build_model(instance)
        
        model.setParam('NodeLimit', GRB.INFINITY)
        model.setParam('TimeLimit', 30*60)
        model.update()
        model.optimize()

        if model.Status == GRB.Status.OPTIMAL:
            return (model.Runtime, "OPTIMAL", model.NodeCount, model.ObjVal, None)
        elif model.Status == GRB.Status.INFEASIBLE:
            return (model.Runtime, "INFEASIBLE", model.NodeCount, None, None)
        elif model.Status == GRB.Status.TIME_LIMIT:
            return (model.Runtime, "TIME_LIMIT_>30min", model.NodeCount, model.ObjVal, model.ObjBound)
        else:
            raise RuntimeError(f"optimize_model : unexpected status : {model.Status}")
    

    @staticmethod
    def optimize_model_at_root_node(instance):

        start = perf_counter()
        model, _ = ModelSolver.build_model(instance)
        end = perf_counter()
        
        start = perf_counter()
        try:
            model = model.presolve()
        except Exception as e:
            return str(e).replace(" ", "")
        end = perf_counter()
        model = model.relax()
        presolve_time = end-start
        
        model.optimize()

        if model.Status == GRB.Status.OPTIMAL:
            return (presolve_time+model.Runtime, "OPTIMAL", model.ObjVal)
        elif model.Status == GRB.Status.INFEASIBLE:
            return (presolve_time+model.Runtime, "INFEASIBLE", None)
        else:
            raise RuntimeError(f"optimize_model_at_root_node : unexpected status : {model.Status}")
        

    # used in subMIPing
    @staticmethod
    def optimize_model_restricted(instance, complete_fixing_history, app_sizes, start_R):

        model, x = ModelSolver.build_model(instance)
         
        N = len(app_sizes)
        shift = [0] * N
        for i in range(1,N):
            shift[i] = sum(app_sizes[:i])
        
        for (n,u,i) in complete_fixing_history:
            model.addConstr(
                x[u+shift[n],i] == 1,
                name=f'fixed_app_{n}_microservice_{u}'
            )
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                t = perf_counter() - start_R #+ model.cbGet(GRB.Callback.RUNTIME)
                print(f"new solution found: z = {obj} , time = {t}")

        model.setParam('NodeLimit', GRB.INFINITY)
        model.setParam('TimeLimit', 10*60)
        model.update()
        model.optimize(callback)

        if model.Status == GRB.Status.OPTIMAL:
            return (model.Runtime, "OPTIMAL", model.NodeCount, model.ObjVal, None)
        elif model.Status == GRB.Status.INFEASIBLE:
            return (model.Runtime, "INFEASIBLE", model.NodeCount, None, None)
        elif model.Status == GRB.Status.TIME_LIMIT:
            return (model.Runtime, "TIME_LIMIT_>10min", model.NodeCount, model.ObjVal, model.ObjBound)
        else:
            raise RuntimeError(f"optimize_model : unexpected status : {model.Status}")