from HSICubeBuilder import HSIImageHandler, CoordSys
from HSICubeBuilder import HSICubeTransformImageExtractor as CTE
import numpy as np

augment_variant = True
augment_transform = True

def applyVariants(HSI_IH):
	I = CTE.getIdentity()
	HSI_IH.addTransformExtractor(CTE.addRotation(I, np.deg2rad(15)))
	HSI_IH.addTransformExtractor(CTE.addRotation(I, np.deg2rad(30)))
	HSI_IH.addTransformExtractor(CTE.addRotation(I, np.deg2rad(45)))
	HSI_IH.addTransformExtractor(CTE.addRotation(I, np.deg2rad(60)))
	HSI_IH.addTransformExtractor(CTE.addRotation(I, np.deg2rad(75)))

	HSI_IH.addTransformExtractor(CTE.addScale(I, 1.5, 1.5))	
	HSI_IH.addTransformExtractor(CTE.addScale(I, 0.7, 0.7))
	HSI_IH.addTransformExtractor(CTE.addScale(I, 1.5, 1.0))
	HSI_IH.addTransformExtractor(CTE.addScale(I, 1.0, 1.5))
