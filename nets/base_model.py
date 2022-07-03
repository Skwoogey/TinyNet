# basic model

Name = 'base_model'
window_size = 11
window_depth = 20
ghost_ratio = 2
conv_structure = [
	#shape (11, 11, 20, 1)
	('conv', 8, (3, 3, 7), 0),
	#shape (9, 9, 14, 8)
	('ghost', 16, (3, 3, 5), ghost_ratio),
	#shape (7, 7, 10, 16)
	('ghost', 32, (3, 3, 3), ghost_ratio),
	#shape (5, 5, 8, 32)
	('ghost', 64, (3, 3, 3), ghost_ratio)
	#shape (3, 3, 6, 64)
]

dense_structure = [
	#shape (3456)
	(256, 'relu', 0.3),
	#shape (256)
	(128, 'relu', 0.4),
	#shape (128)
]

total_epochs = 50
def scheduler(epoch):
	if epoch < 40:
		return 0.001
	else:
		return  0.0001