import numpy as np

from exceptions import InfeasibleOnBuilding

        
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

        ins.R = {'bandwidth', 'core'}  
        ins.S = {'has_gpu', 'latency', 'has_camera'}
        ins.K = {'bandwidth', 'core', 'has_gpu', 'latency', 'has_camera'}

        (
            ins.Q_nodes_R_core, 
            Q_nodes_S_has_camera, 
            Q_nodes_S_has_gpu, 
            ins.Q_links_R_bandwidth, 
            Q_links_S_latency 
        ) = Instance.build_network_rp_availability_from_file(
            network_rp_filename, 
            data= (ins.I, ins.A)
        )

        (
            ins.q_microservices_R_core, 
            q_microservices_S_has_camera, 
            q_microservices_S_has_gpu, 
            ins.q_connections_R_bandwidth, 
            q_connections_S_latency
        ) = Instance.build_app_rp_consumption_from_file(
            app_rp_filename, 
            data= (ins.T, ins.D)
        )

        (
            ins.b_microservices_zero, 
            ins.b_microservices_one,
            ins.b_connections_zero_not_implied, 
            ins.b_connections_one, ins.b_connections_one_actual
        ) = Instance.build_b_coefficients(
            data= (
                ins.I, ins.T, ins.D, ins.P,
                ins.Q_nodes_R_core, ins.Q_links_R_bandwidth, 
                ins.q_microservices_R_core, ins.q_connections_R_bandwidth,
                Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
                q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_S_latency
            )
        ) # can raise InfeasibleOnBuilding exception
        
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
            
            assert len(c) == I
            

            # for any i ∈ I and j ∈ I :
            #       P[i][j] is a list contains the routing path from network node i to network node j
            #       ex. P[i][j] could be [(i,a), (a,b), (b,c), (c,j)]

            
            P = [[None] * I for _ in range(I)] # IxI matrix

            for line in f:
                endpoints, path = line.strip().split(" : ")
                src, dest = map(int, endpoints.split())
                path = list(map(int, path.split()))
                P[src][dest] = tuple([link for link in zip(path, path[1:])])

            # sanity check for paths
            """
            for i in range(I):
                for j in range(I):
                    path = P[i][j]
                    for idx, link in enumerate(path):
                        assert link in A
                        assert idx != 0 or path[idx][0] == i
                        assert idx != (len(path)-1) or path[idx][1] == j
                        assert idx == 0 or path[idx-1][1] == path[idx][0]
            """

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
    def build_network_rp_availability_from_file(filename, data):

        I,A = data

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
            
            assert len(Q_nodes_R_core) == I
            
            ###
            
            assert f.readline().strip() == "has_camera"
            
            Q_nodes_S_has_camera = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            """
            assert len(Q_nodes_S_has_camera) == I
            assert set(Q_nodes_S_has_camera).issubset({0,1}) # binary property
            """
            ###
            
            assert f.readline().strip() == "has_gpu"
            
            Q_nodes_S_has_gpu = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            """
            assert len(Q_nodes_S_has_gpu) == I
            assert set(Q_nodes_S_has_gpu).issubset({0,1}) # binary property
            """
            ###
            
            assert f.readline().strip() == "bandwidth"
            
            Q_links_R_bandwidth = np.full((I, I), -1, dtype=np.int32)

            for _ in range(len(A)//2):
                link, x = f.readline().strip().split()
                i,j = map(int, link.split(','))
                assert (i,j) in A
                Q_links_R_bandwidth[i,j] = int(x)
                Q_links_R_bandwidth[j,i] = Q_links_R_bandwidth[i,j]
            
            """
            # sanity check for Q_links_R_bandwidth
            for (l,m) in A:
                assert Q_links_R_bandwidth[l,m] >= 0
                assert Q_links_R_bandwidth[m,l] >= 0
                # disponibilità simmetriche
                assert Q_links_R_bandwidth[l,m] == Q_links_R_bandwidth[m,l]
            """
            ###
            
            assert f.readline().strip() == "latency"
            
            Q_links_S_latency = np.full((I, I), -1, dtype=np.int32)

            for _ in range(len(A)//2):
                link, x = f.readline().strip().split()
                i,j = map(int, link.split(','))
                assert (i,j) in A
                Q_links_S_latency[i,j] = int(x)
                Q_links_S_latency[j,i] = Q_links_S_latency[i,j]
            
            """
            # sanity check for Q_links_S_latency
            for (l,m) in A:
                assert Q_links_S_latency[l,m] >= 0
                assert Q_links_S_latency[m,l] >= 0
                # disponibilità simmetriche
                assert Q_links_S_latency[l,m] == Q_links_S_latency[m,l]
            """

        return Q_nodes_R_core, Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_R_bandwidth, Q_links_S_latency
    
    
    @staticmethod
    def build_app_rp_consumption_from_file(filename, data):

        T,D = data

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
            
            assert len(q_microservices_R_core) == T
            
            ###
            
            assert f.readline().strip() == "has_camera"
            
            q_microservices_S_has_camera = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            """
            assert len(q_microservices_S_has_camera) == T
            assert set(q_microservices_S_has_camera).issubset({0,1}) # binary property
            """
            ###
            
            assert f.readline().strip() == "has_gpu"
            
            q_microservices_S_has_gpu = np.array(
                list(map(int, f.readline().strip().split())), 
                dtype=np.int32
            )
            
            """
            assert len(q_microservices_S_has_gpu) == T
            assert set(q_microservices_S_has_gpu).issubset({0,1}) # binary property
            """
            ###
            
            assert f.readline().strip() == "bandwidth"

            q_connections_R_bandwidth = np.full((T, T), -1, dtype=np.int32)

            for _ in range(len(D)):
                link, x = f.readline().strip().split()
                u,v = map(int, link.split(','))
                """assert (u,v) in D"""
                q_connections_R_bandwidth[u,v] = int(x)
            
            """
            # sanity check for q_connections_R_bandwidth
            for (u,v) in D:
                assert q_connections_R_bandwidth[u,v] >= 0
            """
            ###
            
            assert f.readline().strip() == "latency"

            q_connections_S_latency = np.full((T, T), -1, dtype=np.int32)

            for _ in range(len(D)):
                link, x = f.readline().strip().split()
                u,v = map(int, link.split(','))
                assert (u,v) in D
                q_connections_S_latency[u,v] = int(x)
            
            """
            # sanity check for q_connections_S_latency
            for (u,v) in D:
                assert q_connections_S_latency[u,v] >= 0
            """

        return q_microservices_R_core, q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_R_bandwidth, q_connections_S_latency
    
   
    @staticmethod
    def build_b_coefficients(data):

        (   
            I, T, D, P,
            Q_nodes_R_core, Q_links_R_bandwidth, 
            q_microservices_R_core, q_connections_R_bandwidth,
            Q_nodes_S_has_camera, Q_nodes_S_has_gpu, Q_links_S_latency,
            q_microservices_S_has_camera, q_microservices_S_has_gpu, q_connections_S_latency
        ) = data

        # b_microservices[u][i]         = 0   if the resource requirement q_microservices[(u,k)]          
        #                                     alone exceeds the availability Q_nodes[(i,k)] for some k ∈ K
        #                                           if u ∈ T && i ∈ I
        #                               = 1   otherwise

        b_microservices = np.zeros((T, I), dtype=np.int32)
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
                    b_microservices[u,i] = 1
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

        """
        for u in range(T):
           assert np.sum(b_microservices[u]) == I-len(b_microservices_zero[u])
        """

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

        b_connections_zero = [[None] * T for _ in range(T)]
        b_connections_zero_not_implied = [[None] * T for _ in range(T)]
        b_connections_one = [[None] * T for _ in range(T)]
        b_connections_one_actual = [[None] * T for _ in range(T)]

        for (u,v) in D:
            b_connections_zero[u][v] = []
            b_connections_zero_not_implied[u][v] = []
            b_connections_one[u][v] = set() # lo uso nell'euristica
            b_connections_one_actual[u][v] = []

            for i in range(I):
                for j in range(I):

                    is_zero = False
                    if q_connections_R_bandwidth[u,v] > v_cum_bandwidth[i,j]:
                        is_zero = True
                    if q_connections_S_latency[u,v] < v_cum_latency[i,j]:
                        is_zero = True

                    if is_zero:
                        b_connections_zero[u][v].append((i,j))
                    else:
                        b_connections_one[u][v].add((i,j))

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

            b_connections_zero[u][v] = tuple(b_connections_zero[u][v])
            b_connections_zero_not_implied[u][v] = tuple(b_connections_zero_not_implied[u][v])
            b_connections_one[u][v] = tuple(b_connections_one[u][v])
            b_connections_one_actual[u][v] = tuple(b_connections_one_actual[u][v])
                    
        # (i,j) ∈ b_connections_zero[u,v] : you cannot map u to i and v to j, 
        #                                   resources on P[i][j] are not enough
        #                               
        # (i,j) ∈  b_connections_one[u,v] : you can map u to i and v to j, 
        #                                   resources on P[i][j] are enough

        """
        # sanity check
        for (u,v) in D:
            assert b_connections_zero[u][v] is not None
            assert b_connections_one[u][v] is not None
            assert len(set(b_connections_one[u][v] + b_connections_zero[u][v])) == I*I
            assert len(set(b_connections_one[u][v]) & set(b_connections_zero[u][v])) == 0
            assert set(b_connections_zero_not_implied[u][v]).issubset(set(b_connections_zero[u][v]))
            assert set(b_connections_one_actual[u][v]).issubset(set(b_connections_one[u][v]))
            for i in range(I):
                for j in range(I):
                    assert (i,j) in b_connections_zero[u][v] or (i,j) in b_connections_one[u][v]
        """

        return b_microservices_zero, b_microservices_one, b_connections_zero_not_implied, b_connections_one, b_connections_one_actual