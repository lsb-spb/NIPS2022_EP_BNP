from .resnet import resnet20  

def get_model(model):
    if model == 'resnet20':
        return resnet20()   
    else:
        raise ValueError(f"Unknown model: {model}")