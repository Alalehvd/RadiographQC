# Radiograph QC — Hold-out Evaluation
_Generated: 2025-10-08T18:37:20_

## Exposure Model
- Train/Test sizes: 121 / 31
- Test positive rate: 0.677
- AUC: **1.000**
- Average Precision: **1.000**
- Brier score: 0.0022

### Confusion Matrix
![cm](exposure_cm.png)

### ROC Curve
![roc](exposure_roc.png)

### Precision–Recall Curve
![pr](exposure_pr.png)

### Calibration Curve
![cal](exposure_calibration.png)

### Top Feature Importances
![fi](exposure_feature_importance.png)

### Classification Report
```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        10
           1      1.000     1.000     1.000        21

    accuracy                          1.000        31
   macro avg      1.000     1.000     1.000        31
weighted avg      1.000     1.000     1.000        31
```

## Quality Model
- Train/Test sizes: 121 / 31
- Test positive rate: 0.774
- AUC: **1.000**
- Average Precision: **1.000**
- Brier score: 0.0027

### Confusion Matrix
![cm](quality_cm.png)

### ROC Curve
![roc](quality_roc.png)

### Precision–Recall Curve
![pr](quality_pr.png)

### Calibration Curve
![cal](quality_calibration.png)

### Top Feature Importances
![fi](quality_feature_importance.png)

### Classification Report
```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000         7
           1      1.000     1.000     1.000        24

    accuracy                          1.000        31
   macro avg      1.000     1.000     1.000        31
weighted avg      1.000     1.000     1.000        31
```

