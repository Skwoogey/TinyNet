# basic model

Name = 'base_model'
window_size = 15
window_depth = 20
ghost_ratio = 2
filter_mul = 1
conv_structure = [
	#shape (11, 11, 30, 1)
	('conv', 8 * filter_mul, (3, 3, 3), 0, 'valid'),
    #('conv', 8 * filter_mul, (1, 1, 3), 0, 'valid'),
	('conv', 8 * filter_mul, (3, 3, 3), 0, 'valid'),

	#shape (9, 9, 24, 32)
	('ghost', 16 * filter_mul, (3, 3, 3), 2, 'valid'),
    #('conv', 16 * filter_mul, (1, 1, 3), 2, 'valid'),
	('ghost', 16 * filter_mul, (3, 3, 3), 2, 'valid'),
	#('conv', 16, (1, 1, 3), 0),
	#shape (7, 7, 18, 64)
	#('ghost', 32, (1, 1, 3), 4),
    ('conv', 32 * filter_mul, (1, 1, 3), 4, 'valid'),
    #('ghost', 32 * filter_mul, (1, 1, 3), 4, 'valid'),
	('ghost', 32 * filter_mul, (3, 3, 3), 4, 'valid'),
	#shape (5, 5, 12, 128)
	#('ghost', 64 * filter_mul, (1, 1, 7), 8, 'same'),
    ('conv', 32 * filter_mul, (1, 1, 3), 4, 'valid'),
    ('ghost', 32 * filter_mul, (3, 3, 3), 4, 'valid'),
	#shape (3, 3, 8, 128)
    #('ghost', 64 * filter_mul, (1, 1, 7), 8, 'same'),
    #('ghost', 64 * filter_mul, (1, 1, 3), 8, 'valid'),
	('ghost', 64 * filter_mul, (3, 3, 3), 8, 'valid'),
	#shape (3, 3, 4, 128)
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

'''
# adam optimizer

# bn + no dropout + 10 samples
# CustomFast3DCNN.py -i datasets\\Salinas\\iPCA\\Salinas_corrected_20bands.mat -gt datasets\\Salinas\Salinas_gt.mat -sp models\\Salinas\\ -m nets\\7w20d_btlneck.py -bs 20 -ds 5 10 85 -a
# CustomFast3DCNN.py -i datasets\\Salinas\\iPCA\\Salinas_corrected_20bands.mat -gt datasets\\Salinas\Salinas_gt.mat -sp models\\Salinas\\ -m nets\\7w20d_btlneck.py -bs 20 -ds 5 50 45 -a

Name = 'base_model'
window_size = 7
window_depth = 20
ghost_ratio = 2
conv_structure = [
	#shape (7, 7, 20, 1)
	('conv', 8, (3, 3, 7), 0),
	#shape (5, 5, 14, 8)
	('ghost', 16, (3, 3, 7), 2),
	#shape (3, 3, 8, 16)
	('ghost', 32, (3, 3, 7), 4),
	#shape (1, 1, 4, 32)
]

dense_structure = [
	#shape (3456)
	(64, 'relu', 0.0),
	#shape (256)
	(64, 'relu', 0.0),
	#shape (128)
]

total_epochs = 100
def scheduler(epoch):
	lr = None
	if epoch < 60:
		lr = 0.001
	elif epoch <= 120:
		lr = 0.0001
	elif epoch <= total_epochs:
		lr = 5e-5
	print(lr)
	return lr
'''