from HSICubeBuilder import HSIImageHandler, CoordSys

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