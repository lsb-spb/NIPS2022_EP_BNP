from .resnet import resnet20 # Ensure your file is named resnet.py

def get_model(name, num_classes, norm_layer=None):
    if name == 'resnet20':
        if norm_layer:
            # This allows EP/BNP to inject their custom BatchNorm layers
            return resnet20(num_classes=num_classes, norm_layer=norm_layer)
        return resnet20(num_classes=num_classes)
    else:
        # Give a specific error message so we know what's missing
        raise ValueError(f"Model {name} not found. Check models/__init__.py")