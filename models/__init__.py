from .mesh_classifier import ClassifierModel
''', Mesh_BaseEncoder_for_SimCLR'''

def create_model(opt):
    model = ClassifierModel(opt)
    return model

'''def create_base_encoder(opt):
    base_encoder = Mesh_BaseEncoder_for_SimCLR(opt)
    return base_encoder'''