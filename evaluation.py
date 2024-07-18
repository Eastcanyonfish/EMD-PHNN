from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error
import numpy as np

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape = (abs(y_predict - y_test) / y_test).mean()*100
    r_2 = r2_score(y_test, y_predict)
    msle = mean_squared_log_error(y_test, y_predict)*100
    MedAE = median_absolute_error(y_test, y_predict)
    TIC = (np.sqrt(mean_squared_error(y_test, y_predict))) / \
        (np.sqrt((y_test**2).mean())+np.sqrt((y_predict).mean()))
    return ['mse: %.6f' % mse, 'rmse: %.6f' % rmse, 'mae: %.6f' % mae, 'mape: %.6f' % mape,
            'msle: %.6f' % msle, 'TIC: %.6f' % TIC, 'r2: %.6f' % r_2]
