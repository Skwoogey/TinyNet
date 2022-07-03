# basic model

Name = 'base_model'
window_size = 5
window_depth = 20
ghost_ratio = 2
conv_structure = [
	#shape (5, 5, 20, 1)

	#('conv', 8, (1, 1, 3), 0),
	('conv', 8, (1, 1, 3), 0, 'valid'),
	('conv', 8, (1, 1, 3), 0, 'valid'),
	('conv', 8, (1, 1, 3), 2, 'valid'),
	('conv', 8, (3, 3, 3), 2, 'valid'),
	#shape (3, 3, 12, 16)
	('ghost', 16, (1, 1, 3), 2, 'valid'),
	('ghost', 16, (1, 1, 3), 2, 'valid'),
	('ghost', 16, (1, 1, 3), 2, 'valid'),
	('ghost', 16, (3, 3, 3), 2, 'valid'),
	#shape (1, 1, 4, 32)
]

dense_structure = [
	0.0,
	#shape (3456)
	#(64, 'relu', 0.0),
	#shape (256)
	#(64, 'relu', 0.0),
	#shape (128)
]

total_epochs = 80*2 //3
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
