class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self, size):
        raise NotImplementedError()
    


class Normal(Variable):
    def __init__(self, name):
        super().__init__(name)
    
    def sample(self, size):
        return np.array([np.random.normal() for _ in range(size)])

