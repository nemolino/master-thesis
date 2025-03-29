import networkx as nx
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

#random.seed(42)

def generate_random_edge_topology(n):

    if n < 2: raise ValueError("n < 2")
    if n > 100: raise ValueError("n > 100")

    with open("waxman_parameters.txt", "r") as f:
        y = None
        for line in f:
            x,y = line.split()
            x = int(x)
            if n == x:
                y = float(y)
                break
        assert y is not None

    while True:
        for _ in range(500):
            G = nx.waxman_graph(n, y, y, domain=(-200,-200,200,200))
            if nx.is_connected(G):
                return G
        y += 0.01


def get_nodes_costs(G, vehicle_nodes, edge_nodes, cloud_nodes):

    vehicle_nodes_costs = dict()
    edge_nodes_costs = dict()
    cloud_nodes_costs = dict()

    for k,v in nx.harmonic_centrality(G,vehicle_nodes).items(): 
        vehicle_nodes_costs[k] = round(v,2)
    for k,v in nx.harmonic_centrality(G,edge_nodes).items(): 
        edge_nodes_costs[k] = round(v,2)
    assert len(cloud_nodes) == 1
    cloud_nodes_costs[cloud_nodes[0]] = 10

    def get_scaling_parameters(values, new_min, new_max):
        if (len(values)) == 1:
            return (new_min + (new_max - new_min) / 2) / list(values)[0], 0
        minimum, maximum = min(values), max(values)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m,b

    # -c-ee-vv

    m, b = get_scaling_parameters(vehicle_nodes_costs.values(), 150, 200)
    for k,v in vehicle_nodes_costs.items():
        vehicle_nodes_costs[k] = int(round(m*v+b, 0))

    m, b = get_scaling_parameters(edge_nodes_costs.values(), 50, 100)
    for k,v in edge_nodes_costs.items():
        edge_nodes_costs[k] = int(round(m*v+b, 0))

    return vehicle_nodes_costs, edge_nodes_costs, cloud_nodes_costs


def create_file_topology_for_network(filename, data):

    (   I, G, 
        vehicle_nodes, edge_nodes, cloud_nodes, paths,
        vehicle_nodes_costs, edge_nodes_costs, cloud_nodes_costs
    ) = data
    
    with open(filename + ".dat", "w") as f:

        f.write(f"{I}\n")

        f.write(" ".join([f"{i},{j}" for (i,j) in G.edges()]) + "\n")

        for i in range(I):
            if i in vehicle_nodes:  f.write(f"{vehicle_nodes_costs[i]}")
            elif i in edge_nodes:   f.write(f"{edge_nodes_costs[i]}")
            elif i in cloud_nodes:  f.write(f"{cloud_nodes_costs[i]}")
            else:                   raise AssertionError('unreachable')
            if i < I-1:
                f.write(" ")
            else:
                assert i==I-1
                f.write("\n")

        for i in sorted(G.nodes):
            for j in sorted(G.nodes):
                f.write(f"{i} {j} : {' '.join(map(str,paths[i][j]))}\n")


def create_file_property_resources_for_network(filename, data, parameters):

    (I, G, vehicle_nodes, edge_nodes, cloud_nodes) = data

    (
        vehicle_min_cores, vehicle_max_cores, edge_min_cores, edge_max_cores, cloud_cores, 
        prob_vehicle_has_camera, prob_vehicle_has_gpu,
        prob_edge_has_camera, prob_edge_has_gpu, 
        min_bandwidth, max_bandwidth, latency
    ) = parameters


    with open(filename + ".dat", "w") as f:
        
        # --------------- resources/properties on nodes

        f.write("core\n")
        for i in range(I):
            if i in vehicle_nodes:  f.write(f"{random.randint(vehicle_min_cores,vehicle_max_cores)}")
            elif i in edge_nodes:   f.write(f"{random.randint(edge_min_cores,edge_max_cores)}")
            elif i in cloud_nodes:  f.write(f"{cloud_cores}")
            else:                   raise AssertionError('unreachable')
            if i < I-1:
                f.write(" ")
        f.write("\n")

        f.write("has_camera\n")
        for i in range(I):
            if i in vehicle_nodes:  f.write(f"{1 if random.random() < prob_vehicle_has_camera else 0}")
            elif i in edge_nodes:   f.write(f"{1 if random.random() < prob_edge_has_camera else 0}")
            else:                   f.write(f"{0}")
            if i < I-1:
                f.write(" ")
        f.write("\n")

        f.write("has_gpu\n")
        for i in range(I):
            if i in vehicle_nodes:  f.write(f"{1 if random.random() < prob_vehicle_has_gpu else 0}")
            elif i in edge_nodes:   f.write(f"{1 if random.random() < prob_edge_has_gpu else 0}")
            else:                   f.write(f"{0}")
            if i < I-1:
                f.write(" ")
        f.write("\n")

        # --------------- resources/properties on edges

        f.write("bandwidth\n")

        for (i,j) in G.edges():
            f.write(f"{i},{j} {random.randint(min_bandwidth,max_bandwidth)}\n")

        f.write(f"latency\n")
        for (i,j) in G.edges():
            f.write(f"{i},{j} {latency}\n")


def generate_network(I, prefix, suffix, rp_count, parameters, verbose=False):

    assert I >= 5   # non si riesce neanche a dividere i nodi in modo sensato con I < 5 

    n = round(I * (2/3))  # edge_nodes_count sono circa 2/3 del totale
    edge_nodes_count = random.randint(round(n - 0.05 * I),round(n + 0.05 * I))
    if verbose:
        print(f"edge_nodes_count random in [{round(n - 0.05 * I)},{round(n + 0.05 * I)}]\n")

    cloud_nodes_count = 1
    vehicle_nodes_count = I - cloud_nodes_count - edge_nodes_count

    assert cloud_nodes_count > 0
    assert vehicle_nodes_count > 0
    assert cloud_nodes_count + edge_nodes_count + vehicle_nodes_count == I

    if verbose:
        print(f"# vehicle nodes = {vehicle_nodes_count}")
        print(f"# edge nodes = {edge_nodes_count} ")
        print(f"# cloud nodes = {cloud_nodes_count}\n")

    edge_nodes = list(range(edge_nodes_count))
    vehicle_nodes = list(range(edge_nodes_count,edge_nodes_count+vehicle_nodes_count))
    cloud_nodes = [I-1]

    if verbose:
        print(f"vehicle nodes : {vehicle_nodes}")
        print(f"edge nodes : {edge_nodes}")
        print(f"cloud nodes : {cloud_nodes}")
        print()

    # genero la topologia dei nodi edge

    G = generate_random_edge_topology(edge_nodes_count)
    colors = ["gold"] * edge_nodes_count
    if verbose:
        plt.figure(1,dpi=200)
        nx.draw(G, pos = nx.spring_layout(G, k=0.8), node_color=colors, edgecolors="black")

    max_degree_of_edge_node = max(list(G.degree(G.nodes)), key=lambda x: x[1])[1] 

    # aggiungo nodi vehicle

    for i in vehicle_nodes:
        G.add_node(i)
        G.add_edge(i,random.choice(edge_nodes))

    colors += (["green"] * vehicle_nodes_count)
    if verbose:
        plt.figure(2,dpi=200)
        nx.draw(G, pos = nx.spring_layout(G, k=0.8), node_color=colors, edgecolors="black")

    # aggiungo nodo cloud

    cloud_node_min_degree = max(max_degree_of_edge_node+1, round(0.5 * edge_nodes_count))
    cloud_node_max_degree = max(cloud_node_min_degree, round(0.75 * edge_nodes_count))
    cloud_node_degree = random.randint(cloud_node_min_degree, cloud_node_max_degree)

    G.add_node(I-1)

    l = edge_nodes[:]
    random.shuffle(l)
    for i in l[:cloud_node_degree]:
        G.add_edge(I-1,i)

    colors += ["tomato"]
    if verbose:
        plt.figure(3,dpi=200)
        nx.draw(G, pos = nx.spring_layout(G, k=0.8), node_color=colors, edgecolors="black")

    # calcolo i cammini fra ogni coppia di nodi

    paths = [[None] * I for _ in range(I)]
    for i in sorted(G.nodes):
        for j in sorted(G.nodes):
            if i <= j:  
                path = list(nx.all_shortest_paths(G,i,j))[0]
            else:       
                path = list(nx.all_shortest_paths(G,i,j))[-1]
            paths[i][j] = path
            assert paths[i][j][0] == i and paths[i][j][-1] == j

    # calcolo i costi dei nodi

    vehicle_nodes_costs, edge_nodes_costs, cloud_nodes_costs = get_nodes_costs(G, vehicle_nodes, edge_nodes, cloud_nodes)
    if verbose:
        v = list(vehicle_nodes_costs.values())
        e = list(edge_nodes_costs.values())
        c = list(cloud_nodes_costs.values())
        print(f"vehicle nodes costs : {v}")
        print(f"edge nodes costs : {e}")
        print(f"cloud nodes costs : {c}")

        min_cost, *_, max_cost = sorted(v + e + c)
        norm = mpl.colors.Normalize(vmin=min_cost, vmax=max_cost, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
        colors = []
        for i in range(I):
            if i in vehicle_nodes:  colors.append(mapper.to_rgba(vehicle_nodes_costs[i]))
            elif i in edge_nodes:   colors.append(mapper.to_rgba(edge_nodes_costs[i]))
            elif i in cloud_nodes:  colors.append(mapper.to_rgba(cloud_nodes_costs[i]))
            else:                   raise AssertionError('unreachable')
        
        plt.figure(4,dpi=200)
        
        nx.draw(G, node_color=colors)

    filename= f"{prefix}network_{I:03d}_{suffix}"

    plt.figure(dpi=200)
    nx.draw(G, pos = nx.spring_layout(G, k=0.8), node_color=colors, edgecolors="black")
    plt.savefig(f'{filename}.png')
    
    create_file_topology_for_network(
        filename, 
        data= (I, G, vehicle_nodes, edge_nodes, cloud_nodes, paths, vehicle_nodes_costs, edge_nodes_costs, cloud_nodes_costs)
    )

    for i in range(rp_count):
        create_file_property_resources_for_network(
            filename= f"{filename}_rp_{i}", 
            data= (I, G, vehicle_nodes, edge_nodes, cloud_nodes), 
            parameters= parameters
        )