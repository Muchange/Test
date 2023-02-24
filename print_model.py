from models.model_ import MCM_Hit
window_size = (4,4)
model = MCM_Hit(n_channels=3, n_classes=1, window_size = window_size)
print(model)