# RadiographQC

RadiographQC is a Streamlit-based prototype for evaluating the technical quality of small animal thoracic radiographs.
It provides automatic exposure assessment, quality scoring, and visual analysis tools built on machine-learning models trained from labeled examples.

⸻

Overview

The goal of this prototype is to explore automated quality control in veterinary radiology.
It estimates exposure, evaluates positioning, sharpness, and collimation,
and visualizes image features to help clinicians understand the technical consistency of radiographs.

⸻

Dataset

The models were trained on a subset of the
**[Mendeley Thoracic Radiograph Dataset](https://data.mendeley.com/datasets/ktx4cj55pn/1)**  
containing **small-animal thoracic X-rays (lateral views)**.

Images were manually labeled for:
	•	Exposure quality
	•	Positioning and collimation
	•	Overall image acceptability

⚠️ Due to limited sample size and lack of anatomical segmentation, results are approximate.
This framework is intended for research purposes only.

⸻

Notes
	•	**The framework shows strong potential for automated QC integration into veterinary imaging workflows.**
	•	Models were trained on a very small public dataset composed mostly of good-quality radiographs with few technical errors,
as the images were originally intended for educational use.
⚠️ As a result, model performance may vary when applied to more diverse clinical data.

⸻

Author

Alaleh Vazifehdoost, DVM
Veterinary Researcher | AI in Diagnostic Imaging
📧 alalehvd@gmail.com
