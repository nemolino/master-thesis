import numpy as np

class InfeasibleOnBuilding(Exception):

    def __init__(self, message="InfeasibleOnBuilding"):
        self.message = message
        super().__init__(self.message)

        
class Instance():


    def __init__(self):
        pass
    

    @staticmethod
    def build(
            network_filename, network_rp_filename, app_filename, app_rp_filename
        ):

        ins = Instance()
        
        ins.I, ins.A, ins.c, ins.P, ins.a2 = Instance.build_network_structure_from_file(network_filename)

        ins.T, ins.D = Instance.build_app_structure_from_file(app_filename)

        # ma questi tre insiemi servono da qualche parte?
        ins.R = {'bandwidth', 'core'}  
        ins.S = {'has_gpu', 'latency', 'has_camera'}
        ins.K = {'bandwidth', 'core', 'has_gpu', 'latency', 'has_camera'}

        (
            ins.Q_nodes_R_core, 
            Q_nodes_S_has_camera, 
            Q_nodes_S_has_gpu, 
            ins.Q_links_R_bandwidth, 
            Q_links_S_latency 
        ) = Instance.build_network_rp_availability_from_file(network_rp_filename, ins)

        (
            ins.q_microservices_R_core, 
            q_microservices_S_has_camera, 
            q_microservices_S_has_gpu, 
            ins.q_connections_R_bandwidth, 
            q_connections_S_latency
        ) = Instance.build_app_rp_consumption_from_file(app_rp_filename, ins)

        data = (
            Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
            q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_S_latency
        )

        (
            ins.b_microservices_zero, 
            ins.b_connections_zero_not_implied, 
            ins.b_connections_one_actual
        ) = Instance.build_b_coefficients(ins, data)
        
        return ins
        

    @staticmethod
    def build_network_structure_from_file(filename):

        ### file format 
        #      line 0 : number of nodes
        #      line 1 : list of links
        #      line 2 : cost of the nodes
        # other lines : "<src> <dest> : <nodes of the path from src to dest>"

        with open(filename) as f:
            
            # I is the number of network nodes
            I = int(f.readline().strip())

            # A is the set of links (i,j) between pairs of network nodes 
            # (all the links are bidirectional)
            A = set()
            for link in f.readline().strip().split():
                i,j = tuple(map(int, link.split(',')))
                A.add((i,j))
                A.add((j,i))

            # c[i] = value score of node i ∈ I
            c = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
        
            

            # for any i ∈ I and j ∈ I :
            #       P[i][j] is a list contains the routing path from network node i to network node j
            #       ex. P[i][j] could be [(i,a), (a,b), (b,c), (c,j)]

            
            P = [[None] * I for _ in range(I)] # IxI matrix

            for line in f:
                endpoints, path = line.strip().split(" : ")
                src, dest = map(int, endpoints.split())
                path = list(map(int, path.split()))
                P[src][dest] = tuple([link for link in zip(path, path[1:])])

            
            # a[(i,j),(l,m)] = 1 if (l,m) ∈ P[i][j]
            #                = 0 otherwise
            # non serve memorizzare tutta la struttura
            
            # helper
            # a2[(l,m)] = ( (i,j) | (l,m) ∈ P[i][j] )

            a2 = [[None] * I for _ in range(I)]
            for (l,m) in A:
                a2[l][m] = []
            for i in range(I):
                for j in range(I):
                    for (l,m) in P[i][j]:
                        a2[l][m].append((i,j))
            for (l,m) in A:
                a2[l][m] = tuple(a2[l][m])

            """
            # sanity check for a2
            
            assert set(a2.keys()) == A
            total_number_of_arcs_calculated_from_P = 0
            for i in range(I):
                for j in range(I):
                    total_number_of_arcs_calculated_from_P += len(P[i][j])
            total_number_of_arcs_calculated_from_a2 = 0
            for k,v in a2.items():
                total_number_of_arcs_calculated_from_a2 += len(v)
            assert total_number_of_arcs_calculated_from_P == total_number_of_arcs_calculated_from_a2
            """

        return I,A,c,P,a2

    
    @staticmethod
    def build_app_structure_from_file(filename):

        ### file format 
        # line 0 : number of microservices
        # line 1 : list of links

        with open(filename) as f:
            
            # T is the number of microservices in the app
            T = int(f.readline().strip())

            # D is the set of communication depencies (u,v) between pairs of microservices
            D = set()
            for link in f.readline().strip().split():
                u,v = tuple(map(int, link.split(',')))
                D.add((u,v))
        
        return T,D
    
    
    @staticmethod
    def build_network_rp_availability_from_file(filename, ins):

        I,A = ins.I, ins.A

        ### file format 
        #   core
        #   space-separated availabilities of resource core on nodes
        #   has_camera
        #   space-separated availabilities of property has_camera on nodes
        #   has_gpu
        #   space-separated availabilities of property has_gpu on nodes
        #   bandwidth
        #   for each network link (i,j):
        #       i,j <availability of the resource bandwidth on link (i,j)>
        #   latency
        #   for each network link (i,j):
        #       i,j <availability of the property latency on link (i,j)>

        with open(filename) as f:
            
            assert f.readline().strip() == "core"

            Q_nodes_R_core = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            
            
            ###
            
            assert f.readline().strip() == "has_camera"
            
            Q_nodes_S_has_camera = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            
            
            ###
            
            assert f.readline().strip() == "has_gpu"
            
            Q_nodes_S_has_gpu = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            
            
            ###
            
            assert f.readline().strip() == "bandwidth"
            
            Q_links_R_bandwidth = np.full((I, I), -1, dtype=np.int32)

            for _ in range(len(A)):
                link, x = f.readline().strip().split()
                i,j = map(int, link.split(','))
                Q_links_R_bandwidth[i,j] = int(x)
            
            ###
            
            assert f.readline().strip() == "latency"
            
            Q_links_S_latency = np.full((I, I), -1, dtype=np.int32)

            for _ in range(len(A)):
                link, x = f.readline().strip().split()
                i,j = map(int, link.split(','))
                Q_links_S_latency[i,j] = int(x)
            
            
        return Q_nodes_R_core, Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_R_bandwidth, Q_links_S_latency
    
    
    @staticmethod
    def build_app_rp_consumption_from_file(filename, ins):

        T,D = ins.T, ins.D

        ### file format 
        #   core
        #   space-separated consumptions of resource core by microservices
        #   has_camera
        #   space-separated consumptions of property has_camera by microservices
        #   has_gpu
        #   space-separated consumptions of property has_gpu by microservices
        #   bandwidth
        #   for each communication dependency (u,v):
        #       u,v <consumption of resource bandwidth on link (u,v)>
        #   latency
        #   for each communication dependency (u,v):
        #       u,v <consumption of property latency on link (u,v)>

        with open(filename) as f:
            
            assert f.readline().strip() == "core"
            
            q_microservices_R_core = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            
            
            ###
            
            assert f.readline().strip() == "has_camera"
            
            q_microservices_S_has_camera = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            
            
            ###
            
            assert f.readline().strip() == "has_gpu"
            
            q_microservices_S_has_gpu = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            ###
            
            assert f.readline().strip() == "bandwidth"

            q_connections_R_bandwidth = np.full((T, T), -1, dtype=np.int32)

            for _ in range(len(D)):
                link, x = f.readline().strip().split()
                u,v = map(int, link.split(','))
                
                q_connections_R_bandwidth[u,v] = int(x)
            
            
            
            ###
            
            assert f.readline().strip() == "latency"

            q_connections_S_latency = np.full((T, T), -1, dtype=np.int32)

            for _ in range(len(D)):
                link, x = f.readline().strip().split()
                u,v = map(int, link.split(','))
                
                q_connections_S_latency[u,v] = int(x)
            
            
            
        return q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_R_bandwidth, q_connections_S_latency
    
   
    @staticmethod
    def build_b_coefficients(ins, data):

        (
            I, T, D, P,
            Q_nodes_R_core, Q_links_R_bandwidth, 
            q_microservices_R_core, q_connections_R_bandwidth
        ) = (
            ins.I, ins.T, ins.D, ins.P,
            ins.Q_nodes_R_core, ins.Q_links_R_bandwidth, 
            ins.q_microservices_R_core, ins.q_connections_R_bandwidth
        )

        (
            Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
            q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_S_latency
        ) = data

        # b_microservices[u][i]         = 0   if the resource requirement q_microservices[(u,k)]          
        #                                     alone exceeds the availability Q_nodes[(i,k)] for some k ∈ K
        #                                           if u ∈ T && i ∈ I
        #                               = 1   otherwise

       
        b_microservices_zero = []
        b_microservices_one = []

        for u in range(T):
            b_microservices_zero.append([])
            b_microservices_one.append([])
            infeasible = True
            for i in range(I):
                if (
                    q_microservices_R_core[u] <= Q_nodes_R_core[i] and
                    q_microservices_S_has_camera[u] <= Q_nodes_S_has_camera[i] and
                    q_microservices_S_has_gpu[u] <= Q_nodes_S_has_gpu[i]
                ):
                    infeasible = False
                    b_microservices_one[-1].append(i)
                else:
                    b_microservices_zero[-1].append(i)
            b_microservices_zero[-1] = tuple(b_microservices_zero[-1])
            b_microservices_one[-1] = tuple(b_microservices_one[-1])
            # checking that we can place micro-service u in at least one node
            if infeasible:
                raise InfeasibleOnBuilding() # there is a microservice you cannot place
        b_microservices_zero = tuple(b_microservices_zero)
        b_microservices_one = tuple(b_microservices_one)

        

        # v_cum_bandwidth[i,j] =    cumulative consumption of bandwidth
        #                           along the routing path from i to j   
        #                           i ∈ I , j ∈ I
        
        #   v_cum_latency[i,j] =    cumulative consumption of latncy
        #                           along the routing path from i to j   
        #                           i ∈ I , j ∈ I

        v_cum_bandwidth = np.zeros((I, I), dtype=np.int32)
        v_cum_latency = np.zeros((I, I), dtype=np.int32) 

        def f_bandwidth(xs):
            return min(xs) if len(xs) > 0 else 1_000_000_000
        
        def f_latency(xs):
            return sum(xs)
        
        for i in range(I):
            for j in range(I):
                xs = [Q_links_R_bandwidth[l,m] for (l,m) in P[i][j]]
                v_cum_bandwidth[i,j] = f_bandwidth(xs)
                xs = [Q_links_S_latency[l,m] for (l,m) in P[i][j]]
                v_cum_latency[i,j] = f_latency(xs)
        
        # bandwidth on the path between the nodes that host u and v 
        # must be >= q_connections_R_bandwidth[u,v]

        # latency on the path between the nodes that host u and v 
        # must be <= q_connections_S_latency[u,v]

        
        b_connections_zero_not_implied = [[None] * T for _ in range(T)]
        
        b_connections_one_actual = [[None] * T for _ in range(T)]

        for (u,v) in D:
            
            b_connections_zero_not_implied[u][v] = []
            
            b_connections_one_actual[u][v] = []

            

            for i in b_microservices_one[u]:
                for j in b_microservices_one[v]:

                    is_zero = False
                    if q_connections_R_bandwidth[u,v] > v_cum_bandwidth[i,j]:
                        is_zero = True
                    if q_connections_S_latency[u,v] < v_cum_latency[i,j]:
                        is_zero = True

                    if is_zero:
                        b_connections_zero_not_implied[u][v].append((i,j))
                    else:
                        b_connections_one_actual[u][v].append((i,j))

            
            b_connections_zero_not_implied[u][v] = tuple(b_connections_zero_not_implied[u][v])
            
            b_connections_one_actual[u][v] = tuple(b_connections_one_actual[u][v])
                    
        # (i,j) ∈ b_connections_zero[u,v] : you cannot map u to i and v to j, 
        #                                   resources on P[i][j] are not enough
        #                               
        # (i,j) ∈  b_connections_one[u,v] : you can map u to i and v to j, 
        #                                   resources on P[i][j] are enough

        
        
        
        return b_microservices_zero, b_connections_zero_not_implied, b_connections_one_actual


"""
    MEMBERS_DESCRIPTION = 
self.{ 
    I A P a c
    T D
    R S K
    Q_nodes Q_links
    q_microservices q_connections
    b_microservices b_connections
}

 ---------------------------------------------------------------------------
| NETWORK STRUCTURE                                                         |
 ---------------------------------------------------------------------------

I is the set of network nodes (numbered starting from 1) 

A is the set of links (i,j) between pairs of network nodes (all the links are bidirectional)

for any i ∈ I and j ∈ I :
      P[i][j] is a list that contains the routing path from network node i to network node j
      ex. P[i][j] could be [(i,a), (a,b), (b,c), (c,j)]

a[(i,j),(l,m)]    = 1         if (l,m) ∈ A && (l,m) ∈ P[i][j]
                  = 0         if (l,m) ∈ A && (l,m) ∉ P[i][j]
                  = None      otherwise

c[i]      = value score of node i     if i ∈ I
          = None or IndexError        otherwise

 ---------------------------------------------------------------------------
| MICROSERVICES STRUCTURE                                                   |
 ---------------------------------------------------------------------------

T is the set of microservices to deploy (numbered starting from 1) 

D is the set of communication dependencies (u,v) between pairs of microservices

 ---------------------------------------------------------------------------
| RESOURCES AND PROPERTIES                                                  |
 ---------------------------------------------------------------------------

R is the set of resources     ex. {'bandwidth', 'core'}

S is the set of properties    ex. {'has_camera', 'has_gpu', 'latency'}

K = R + S

 ---------------------------------------------------------------------------
| RESOURCES AND PROPERTIES AVAILABILITY ON THE NETWORK                      |
 ---------------------------------------------------------------------------

Q_nodes[(i,k)]            = availability of resource/property k on node i         
                                      if i ∈ I && k ∈ K
                          = None      otherwise

Q_links[((i,j),k)]        = availability of resource/property k on link (i,j)     
                                      if (i,j) ∈ A && k ∈ K
                          = None      otherwise

 ---------------------------------------------------------------------------
| RESOURCES AND PROPERTIES CONSUMPTION OF THE MICROSERVICES                 |
 ---------------------------------------------------------------------------

q_microservices[(u,k)]    = consumption of resource k by microservice u                   
                                      if u ∈ T && k ∈ K
                          = None      otherwise

q_connections[((u,v),k)]  = consumption (requirement) of resource k by connection (u,v)   
                                      if (u,v) ∈ D && k ∈ K
                          = None      otherwise

 ---------------------------------------------------------------------------
| COEFFICIENTS b                                                            |
 ---------------------------------------------------------------------------

b_microservices[u][i]         = 0   if the resource requirement q_microservices[(u,k)]          
                                    alone exceeds the availability Q_nodes[(i,k)] for some k ∈ K
                                          if u ∈ T && i ∈ I
                              = 1   if ... 
                                    alone does not exceed ...                                  
                                          if u ∈ T && i ∈ I
                              = None or IndexError 
                                          otherwise

b_connections[((u,v),(i,j))]  = 0   if the resource requirement q_connections[((u,v),k)]        
                                    alone exceeds the cumulative consumption of resource k 
                                    along the routing path from i to j for some k ∈ K
                                          if i ∈ I && j ∈ I && (u,v) ∈ D
                              = 1   if ... 
                                    alone does not exceed ...
                                    ...
                                          if i ∈ I && j ∈ I && (u,v) ∈ D
                              = None      
                                          otherwise
"""