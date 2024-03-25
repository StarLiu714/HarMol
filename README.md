# HarMol: AI assisted Recipe Property Prediction Model

## Preliminary Test

### MolSets Dataset: 1077 entries
[molset_smiles_clean.csv](https://github.com/StarLiu714/HarMol/files/14746855/molset_smiles_clean.csv)
[molset_clean_ratio.csv](https://github.com/StarLiu714/HarMol/files/14746787/molset_clean_ratio.csv) 

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
![regression](https://github.com/StarLiu714/HarMol/assets/87756322/05715870-570e-4888-8518-f1b2da22e6bc)
1. Pearson correlation: 0.91999
2. Spearman r: 0.91638


### ChemArr (2023 ACS.Cent.Sci) Dataset

![acs cent sci2023](https://github.com/StarLiu714/HarMol/assets/87756322/32328788-4275-4da1-8ed4-b95e475ba228)


