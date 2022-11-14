import pandas as pd

class LinearRegressionResult:
    def __init__(self):
        self.coefficients: pd.DataFrame = []
        self.normalized_data: pd.DataFrame = None
        self.MAE_metrics = 0
        self.MSE_metrics = 0
        self.RMSE_metrics = 0
    
    def get_coeff(self, index: int):
        col_name = self.coefficients.columns[0]
        series = self.coefficients[col_name].sort_values()
        items_list = list(dict(series).items())
        return items_list[index]
    
    def max_coeff(self):
        return self.get_coeff(-1)
    
    def min_coeff(self) -> pd.Series:
        return self.get_coeff(0)
    
    def print_overview(self):
        print("\n...Regressão Linear Finalizada...\n")
        print("-> Dataset Normalizado usado para treinamento:\n")
        
        _max_col_len = max([len(col) for col in self.normalized_data])
        
        print(' #  ', 'Coluna', ' '*(_max_col_len-6), '  Tipo')
        print('--- ', '------', ' '*(_max_col_len-6), '  ----')
        for (i, col) in enumerate(self.normalized_data):
            _index = str(i+1)
            print(' ' + _index + ' '*(3-len(_index)), col, ' '*(_max_col_len - len(col)), f'- {self.normalized_data[col].dtype}')
            
        print('\n-> Coeficientes:\n')
        print(self.coefficients)
        
        _max = self.max_coeff()
        _min = self.min_coeff()
        _lbl_max = str(_max[0])
        _lbl_min = str(_min[0])
        _max_lbl_len = max([len(_lbl_max), len(_lbl_min)])
        
        print('')
        print(f'Max Coeficiente:  {_lbl_max + " "*(_max_lbl_len-len(_lbl_max))}  {_max[1]}')
        print(f'Min Coeficiente:  {_lbl_min + " "*(_max_lbl_len-len(_lbl_min))}  {_min[1]}')
        
        print('\n-> Métricas de Erro:\n')
        print(f'Mean Absolute Error        (MAE):  {self.MAE_metrics}')
        print(f'Mean Squared Error         (MSE):  {self.MSE_metrics}')
        print(f'Root Mean Squared Error   (RMSE):  {self.RMSE_metrics}')
        print('\n...Fim do Overview...')
        