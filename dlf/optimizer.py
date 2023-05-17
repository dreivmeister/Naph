def SGD_step(model_params, learning_rate):
    # model_params = model.parameters()
    for p in model_params:
        if p != []:
            p.data = p.data - learning_rate*p.grad
        
class Adam:
    def __init__(self,beta1,beta2):
        self.m = []
        self.v = []
        self.beta1 = beta1
        self.beta2 = beta2
        
    def step(self, model_params, learning_rate):
        pass