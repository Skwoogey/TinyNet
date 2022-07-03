from HSICubeBuilder import HSIImageHandler, CoordSys
from HSICubeBuilder import HSICubeTransformImageExtractor as CTE
from HSICubeBuilder import HSICubeSavedTransforms as ST
import numpy as np

augment_variant = True
augment_transform = True

def applyVariants(HSI_IH):
	HSI_IH.addSavedTransforms('ro15.mat')
	HSI_IH.addSavedTransforms('ro30.mat')
	HSI_IH.addSavedTransforms('ro45.mat')
	HSI_IH.addSavedTransforms('ro60.mat')
	HSI_IH.addSavedTransforms('ro75.mat')

	HSI_IH.addSavedTransforms('ro15_sc0.5_0.5.mat')
	HSI_IH.addSavedTransforms('ro30_sc0.5_0.5.mat')
	HSI_IH.addSavedTransforms('ro45_sc0.5_0.5.mat')
	HSI_IH.addSavedTransforms('ro60_sc0.5_0.5.mat')

	HSI_IH.addSavedTransforms('ro60_sc1.0_0.5.mat')
	HSI_IH.addSavedTransforms('ro30_sc1.0_0.5.mat')
	HSI_IH.addSavedTransforms('sc1.0_0.5_ro60.mat')
	HSI_IH.addSavedTransforms('sc1.0_0.5_ro30.mat')


	HSI_IH.addVariant(CoordSys([1, 1]))
	HSI_IH.addVariant(CoordSys([2, 1]))
	HSI_IH.addVariant(CoordSys([2, -1]))
	HSI_IH.addVariant(CoordSys([0, 2]))
	HSI_IH.addVariant(CoordSys([0, 2]))

	HSI_IH.addVariant(CoordSys([0, 1], [1, 1]))
	HSI_IH.addVariant(CoordSys([0, 1], [1, -1]))
	HSI_IH.addVariant(CoordSys([1, 1], [0, 1]))
	HSI_IH.addVariant(CoordSys([1, -1], [0, 1]))
	HSI_IH.addVariant(CoordSys([0, 2], [2, 1]))
	HSI_IH.addVariant(CoordSys([0, 2], [2, -1]))
	HSI_IH.addVariant(CoordSys([2, 1], [0, 2]))
	HSI_IH.addVariant(CoordSys([2, -1], [0, 2]))
