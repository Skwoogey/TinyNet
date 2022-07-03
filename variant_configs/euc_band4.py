from HSICubeBuilder import HSIImageHandler, CoordSys, HSICubePredictSequencer

augment_variant = True
augment_transform = True

def applyVariants(HSI_IH):
	HSI_IH.addVariant(CoordSys([1, 1]))
	HSI_IH.addVariant(CoordSys([2, 1]))
	HSI_IH.addVariant(CoordSys([2, -1]))
	HSI_IH.addVariant(CoordSys([2, 2]))
	HSI_IH.addVariant(CoordSys([0, 2]))
	HSI_IH.addVariant(CoordSys([0, 2], [1, 0]))
	HSI_IH.addVariant(CoordSys([0, 1], [2, 0]))

	HSI_IH.addVariant(CoordSys([3, -1]))
	HSI_IH.addVariant(CoordSys([3, 1]))
	HSI_IH.addVariant(CoordSys([3, -2]))
	HSI_IH.addVariant(CoordSys([3, 2]))
	HSI_IH.addVariant(CoordSys([0, 2], [3, 0]))
	HSI_IH.addVariant(CoordSys([0, 3], [2, 0]))
	HSI_IH.addVariant(CoordSys([0, 1], [3, 0]))
	HSI_IH.addVariant(CoordSys([0, 3], [1, 0]))
	HSI_IH.addVariant(CoordSys([0, 3]))
	HSI_IH.addVariant(CoordSys([3, 3]))

	HSI_IH.addVariant(CoordSys([4, -1]))
	HSI_IH.addVariant(CoordSys([4, 1]))
	HSI_IH.addVariant(CoordSys([4, -2]))
	HSI_IH.addVariant(CoordSys([4, 2]))
	HSI_IH.addVariant(CoordSys([4, -3]))
	HSI_IH.addVariant(CoordSys([4, 3]))
	HSI_IH.addVariant(CoordSys([0, 1], [4, 0]))
	HSI_IH.addVariant(CoordSys([0, 4], [1, 0]))
	HSI_IH.addVariant(CoordSys([0, 2], [4, 0]))
	HSI_IH.addVariant(CoordSys([0, 4], [2, 0]))
	HSI_IH.addVariant(CoordSys([0, 3], [4, 0]))
	HSI_IH.addVariant(CoordSys([0, 4], [3, 0]))
	HSI_IH.addVariant(CoordSys([0, 4]))
	HSI_IH.addVariant(CoordSys([4, 4]))