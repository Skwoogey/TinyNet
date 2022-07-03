# basic model

Name = 'base_model'
window_size = 11
window_depth = 20
ghost_ratio = 2
conv_structure = [
	#shape (11, 11, 20, 1)
	('conv', 8, (3, 3, 7), 0),
	#shape (9, 9, 14, 8)
	('ghost', 16, (3, 3, 5), 2),
	#shape (7, 7, 10, 16)
	('ghost', 32, (3, 3, 3), 4),
	#shape (5, 5, 8, 32)
	('ghost', 64, (3, 3, 3), 8),
	#shape (3, 3, 6, 64)
	('ghost', 128, (3, 3, 3), 16)
	#shape (1, 1, 4, 128)
]

dense_structure = [
	#shape (3456)
	(128, 'relu', 0.3),
	#shape (256)
	(64, 'relu', 0.4),
	#shape (128)
]

total_epochs = 60
def scheduler(epoch):
	if epoch < 40:
		return 0.001
	elif epoch <= total_epochs:
		return  0.0001 + (total_epochs - epoch) * 0.0002