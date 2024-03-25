# HarMol: AI assisted Recipe Property Prediction Model

## Preliminary Test

### MolSets 1077 Dataset

#### Arrhenius Equation re-construct:
   ```python
   def fit_and_predict_conductivity(group_data, temp_to_predict):
     # Transformations
     group_data['1/T'] = 1 / group_data['temperature']
     group_data['log_sigma'] = group_data['conductivity']
     # Linear regression features and target
     X = group_data[['1/T']]
     y = group_data['log_sigma']
     # Fit
     model = LinearRegression()
     model.fit(X, y)
     # Predict
     inverse_temp = 1 / temp_to_predict
     predicted_log_sigma = model.predict([[inverse_temp]])
     predicted_log_sigma_col = predicted_log_sigma[0]
     return predicted_log_sigma_col
 ```

#### XGBoost Regressor Parameters
```python
xg_reg = xgb.XGBRegressor(
 n_estimators=400, objective='reg:squarederror', colsample_bytree=0.3,
 learning_rate=0.05, max_depth=6, alpha=3,
 # tree_method='gpu_hist' if device.type == 'cuda' else 'auto'
 )
```

#### Sample Result
![image](https://github.com/StarLiu714/HarMol/assets/87756322/dc2ee347-579b-4704-a26d-fec3bc3b9a05)
1. Pearson correlation: 0.91999
2. Spearman r: 0.91638
   
