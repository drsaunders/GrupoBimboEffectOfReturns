# Do returns predict future sales? For the GrupoBimbo Inventory Demand competition

I played around a bit in the [Grupo Bimbo Inventory Demand kaggle](https://www.kaggle.com/c/grupo-bimbo-inventory-demand), just to get experience  working with a dataset that was 3.2 GB and 74 million rows, and to submit to my first real Kaggle competition. Although I didn't rank very high, I did one investigation that took a different angle from most people: looking for evidence of a causal effect of one variable on another with a time lag, in this case whether the number of returns for a product predicted whether the client would buy more or not.

It is contained in this 
### [Jupyter notebook: GrupoBimboEffectOfReturns.ipynb](https://github.com/drsaunders/GrupoBimboEffectOfReturns/blob/master/GrupoBimboEffectOfReturns.ipynb)

To rerun it requires that the `train.csv` data file for the competition is in the directory '../input' relative to the notebook.