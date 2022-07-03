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

	HSI_IH.addSavedTransforms('sc15.mat')
	HSI_IH.addSavedTransforms('sc07.mat')
	HSI_IH.addSavedTransforms('sc15_10.mat')
	HSI_IH.addSavedTransforms('sc10_15.mat')
