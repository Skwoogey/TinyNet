# basic model

Name = 'base_model'
window_size = 9
window_depth = 20
ghost_ratio = 2
conv_structure = [
	#shape (9, 9, 20, 1)

	#('conv', 8, (1, 1, 3), 0),
	('conv', 8, (1, 1, 3), 0),
	('conv', 8, (3, 3, 3), 0),
	#shape (7, 7, 14, 8)
	('ghost', 16, (1, 1, 3), 2),
	('ghost', 16, (3, 3, 3), 2),
	#shape (5, 5, 10, 16)
	('ghost', 32, (1, 1, 3), 4),
	('ghost', 32, (3, 3, 3), 4),
	#shape (3, 3, 6, 32)
	('ghost', 64, (1, 1, 3), 8),
	('ghost', 64, (3, 3, 3), 8)
	#shape (1, 1, 4, 64)
]

dense_structure = [
	0.0,
	#shape (3456)
	#(64, 'relu', 0.0),
	#shape (256)
	#(64, 'relu', 0.0),
	#shape (128)
]

total_epochs = 80*2
lr_modifier = 1
def scheduler(epoch):
	lr = None
	if epoch < total_epochs*0.65:
		lr = 0.003 * lr_modifier
	elif epoch <= total_epochs*0.8:
		lr = 0.0006 * lr_modifier
	elif epoch <= total_epochs*0.9:
		lr = 0.00012 * lr_modifier
	else:
		lr = 0.000024 * lr_modifier
	#print(lr)
	return lr
