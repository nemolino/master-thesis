class InfeasibleOnBuilding(Exception):

    def __init__(self, message="InfeasibleOnBuilding"):
        self.message = message
        super().__init__(self.message)


class InfeasiblePricing(Exception):

    def __init__(self, message="InfeasiblePricing"):
        self.message = message
        super().__init__(self.message)


class InfeasibleMaster(Exception):

    def __init__(self, message="InfeasibleMaster"):
        self.message = message
        super().__init__(self.message)