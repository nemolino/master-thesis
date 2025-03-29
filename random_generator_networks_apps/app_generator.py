import networkx as nx
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from math import ceil


def generate_microservices_path(size):

    G = nx.DiGraph()
    G.add_node(0)
    G.add_edges_from([(i,i+1) for i in range(size-1)])

    return G


def generate_microservices_tree(size):
    
    G = nx.DiGraph()
    G.add_nodes_from(range(size))

    for i in range(1,size):
        j = random.randint(0,i-1)
        G.add_edge(i,j)
    
    return G


def divide_microservices_in_a_tree(T, TREE, parameters):

    edge_microservices_count = ceil((T-2) * 0.6)
    vehicle_microservices_count = ceil((T-edge_microservices_count)/2)

    nodes_sorted_by_dist = sorted(list(TREE.nodes),key=lambda x: nx.shortest_path_length(TREE,x,0), reverse=True)
    vehicle_microservices = set(nodes_sorted_by_dist[:vehicle_microservices_count])
    edge_microservices = set(nodes_sorted_by_dist[vehicle_microservices_count:vehicle_microservices_count+edge_microservices_count])
    cloud_microservices = set(nodes_sorted_by_dist[vehicle_microservices_count+edge_microservices_count:])
    
    assert vehicle_microservices | edge_microservices | cloud_microservices == set(range(T))

    return vehicle_microservices, edge_microservices, cloud_microservices


def generate_microservices_graph_from_tree(TREE, T, verbose=False):

    def get_truncated_normal(mean, sd, low, upp):
        return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

    X = get_truncated_normal(mean=round((1/3)*T), sd=T/3, low=1, upp=T)

    if verbose:
        plt.figure(dpi=200)
        
        values = list(map(lambda x: round(x), X.rvs(10000)))

        x = list(range(T+1))
        y = [0] * (T+1)
        for v in values:
            y[v] += 1
        for i in range(T+1):
            y[i] /= 10000
        assert abs(sum(y)-1) < 0.001
        print(x)
        print(y)

        
        plt.bar(x,y,1,color="lightgrey",edgecolor="black")
        plt.xlabel("# added arcs")
        plt.xticks(list(range(1,T+1)))
        plt.title("empirical PMF of the #added arcs")
        plt.show()

    TREE_complement = nx.complement(TREE)
    TREE_complement_edges = [e for e in TREE_complement.edges]

    assert len(TREE.edges) + len(TREE_complement_edges) == T * (T-1)

    # vogliamo che al peggio il # archi totale aumenti di T, cioè che praticamente raddoppi
    # il numero di archi aggiunti è una variabile casuale

    number_of_arcs_to_add = round(X.rvs()) 
    assert 1 <= number_of_arcs_to_add <= T 
    number_of_arcs_to_add = min(number_of_arcs_to_add, len(TREE_complement_edges))

    if verbose:
        print(f"# archi aggiunti : {number_of_arcs_to_add} / {len(TREE_complement_edges)}")

    if number_of_arcs_to_add != 0:
        
        arcs_to_add = random.sample(TREE_complement_edges, number_of_arcs_to_add)
        for u,v in arcs_to_add:
            TREE.add_edge(u,v)

    else:
        print("nothing added") # lascio questo print perchè tanto non dovrebbe capitare mai

    return TREE # actually now is a graph


def create_file_topology_for_app(filename, data):

    T,G = data
    
    with open(filename + ".dat", "w") as f:
        f.write(f"{T}\n")
        f.write(" ".join([f"{u},{v}" for (u,v) in G.edges()]) + "\n")


def create_file_property_resources_for_app(filename, data, parameters):

    (T, G, vehicle_microservices, edge_microservices, cloud_microservices) = data
    (
        vehicle_min_cores, vehicle_max_cores, 
        edge_min_cores, edge_max_cores, 
        cloud_min_cores, cloud_max_cores, 
        prob_vehicle_needs_camera, prob_vehicle_needs_gpu,
        prob_edge_needs_camera, prob_edge_needs_gpu, 
        min_bandwidth, max_bandwidth, 
        get_latency
    ) = parameters
    
    with open(filename + ".dat", "w") as f:

        # --------------- resources/properties on nodes

        f.write("core\n")
        for i in range(T):
            if i in vehicle_microservices:  
                x = random.randint(vehicle_min_cores,vehicle_max_cores)
            elif i in edge_microservices:   
                x = random.randint(edge_min_cores,edge_max_cores)
            elif i in cloud_microservices:  
                x = random.randint(cloud_min_cores,cloud_max_cores)
            else:                           
                raise AssertionError('unreachable')
            f.write(f"{x} ")
        f.write("\n")

        f.write("has_camera\n")
        for i in range(T):
            if i in vehicle_microservices:  
                x = 1 if random.random() < prob_vehicle_needs_camera else 0
            elif i in edge_microservices:  
                x = 1 if random.random() < prob_edge_needs_camera else 0
            else:                           
                x = 0                         
            f.write(f"{x} ")
        f.write("\n")

        f.write("has_gpu\n")
        for i in range(T):
            if i in vehicle_microservices:  
                x = 1 if random.random() < prob_vehicle_needs_gpu else 0
            elif i in edge_microservices:  
                x = 1 if random.random() < prob_edge_needs_gpu else 0
            else:                           
                x = 0   
            f.write(f"{x} ")
        f.write("\n")

        # --------------- resources/properties on edges

        f.write("bandwidth\n")
        for (u,v) in G.edges():
            x = random.randint(min_bandwidth,max_bandwidth)
            f.write(f"{u},{v} {x}\n")

        f.write("latency\n")
        for (u,v) in G.edges():
            f.write(f"{u},{v} {get_latency()}\n")


def create_files(filename, T, G, vehicle_microservices, edge_microservices, cloud_microservices, rp_count, parameters):

    create_file_topology_for_app(
        filename, 
        data= (T, G)
    )

    for i in range(rp_count):

        create_file_property_resources_for_app(
            filename= f"{filename}_rp_{i}", 
            data= (T, G, vehicle_microservices, edge_microservices, cloud_microservices), 
            parameters= parameters
        )


def generate_path_apps(T, prefix, suffix, rp_count, parameters, verbose=False):

    assert 3 <= T <= 20
    assert 1 <= rp_count <= 10

    PATH = generate_microservices_path(size=T)

    edge_microservices_count = ceil((T-2) * 0.6)
    vehicle_microservices_count = ceil((T-edge_microservices_count)/2)
    cloud_microservices_count = T-edge_microservices_count-vehicle_microservices_count

    assert edge_microservices_count >= vehicle_microservices_count
    assert edge_microservices_count >= cloud_microservices_count
    assert vehicle_microservices_count >= cloud_microservices_count

    vehicle_microservices = set(range(vehicle_microservices_count))
    edge_microservices = set(range(vehicle_microservices_count,vehicle_microservices_count+edge_microservices_count))
    cloud_microservices = set(range(vehicle_microservices_count+edge_microservices_count,T))

    assert vehicle_microservices | edge_microservices | cloud_microservices == set(range(T))
    assert vehicle_microservices & edge_microservices & cloud_microservices == set()
    assert len(vehicle_microservices) == vehicle_microservices_count
    assert len(edge_microservices) == edge_microservices_count
    assert len(cloud_microservices) == cloud_microservices_count

    colors = (
        (["green"] * vehicle_microservices_count) +
        (["gold"] * edge_microservices_count) +
        (["tomato"] * cloud_microservices_count)
    )
    
    if verbose:
        print(f"T = {T}\t : {'v' * vehicle_microservices_count} {'e' * edge_microservices_count} {'c' * cloud_microservices_count}")
        print(f"{vehicle_microservices} {edge_microservices} {cloud_microservices}")
        plt.figure(dpi=200)
        nx.draw(PATH, pos=nx.circular_layout(PATH), with_labels=False, node_color=colors, edgecolors="black")

    filename= f"{prefix}path_{T:02d}_{suffix}"

    plt.figure(dpi=200)
    nx.draw(PATH, pos = nx.circular_layout(PATH), with_labels=False, node_color=colors, edgecolors="black")
    plt.savefig(f'{filename}.png')

    create_files(filename, T, PATH, vehicle_microservices, edge_microservices, cloud_microservices, rp_count, parameters)


def generate_tree_apps(T, prefix, suffix, rp_count, parameters, verbose=False):

    assert 3 <= T <= 20
    assert 1 <= rp_count <= 10

    TREE = generate_microservices_tree(size=T)

    threshold1 = 2/3
    threshold2 = 0.2

    vehicle_microservices, edge_microservices, cloud_microservices = divide_microservices_in_a_tree(
        T, TREE, parameters= (threshold1, threshold2)
    )

    assert vehicle_microservices | edge_microservices | cloud_microservices == set(range(T))
    assert vehicle_microservices & edge_microservices & cloud_microservices == set()
    
    colors = [
        "green" if i in vehicle_microservices else (
            "gold" if i in edge_microservices else "tomato"
        )
        for i in range(T)
    ]
    
    if verbose:
        print(f"{vehicle_microservices} {edge_microservices} {cloud_microservices}")
        plt.figure(dpi=200)
        nx.draw(TREE, pos=nx.planar_layout(TREE), with_labels=False, node_color=[colors[node] for node in TREE.nodes], edgecolors="black")

    filename= f"{prefix}tree_{T:02d}_{suffix}"

    plt.figure(dpi=200)
    nx.draw(TREE, pos=nx.planar_layout(TREE), with_labels=False, node_color=[colors[node] for node in TREE.nodes], edgecolors="black")
    plt.savefig(f'{filename}.png')

    create_files(filename, T, TREE, vehicle_microservices, edge_microservices, cloud_microservices, rp_count, parameters)


def generate_graph_apps(T, prefix, suffix, rp_count, parameters, verbose=False):

    assert 3 <= T <= 20
    assert 1 <= rp_count <= 10

    TREE = generate_microservices_tree(size=T)

    threshold1 = 2/3
    threshold2 = 0.2

    vehicle_microservices, edge_microservices, cloud_microservices = divide_microservices_in_a_tree(
        T, TREE, parameters= (threshold1, threshold2)
    )

    assert vehicle_microservices | edge_microservices | cloud_microservices == set(range(T))
    assert vehicle_microservices & edge_microservices & cloud_microservices == set()
    
    GRAPH = generate_microservices_graph_from_tree(TREE, T, verbose)
    
    colors = [
        "green" if i in vehicle_microservices else (
            "gold" if i in edge_microservices else "tomato"
        )
        for i in range(T)
    ]
    
    if verbose:
        print(f"{vehicle_microservices} {edge_microservices} {cloud_microservices}")
        plt.figure(dpi=200)
        nx.draw(GRAPH, pos=nx.planar_layout(GRAPH), with_labels=False, node_color=[colors[node] for node in GRAPH.nodes], edgecolors="black")

    filename= f"{prefix}graph_{T:02d}_{suffix}"

    plt.figure(dpi=200)
    nx.draw(GRAPH, pos=nx.spring_layout(GRAPH), with_labels=False, node_color=[colors[node] for node in GRAPH.nodes], edgecolors="black")
    plt.savefig(f'{filename}.png')

    create_files(filename, T, GRAPH, vehicle_microservices, edge_microservices, cloud_microservices, rp_count, parameters)