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