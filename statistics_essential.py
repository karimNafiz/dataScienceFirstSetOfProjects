import numpy as np 
import pandas as pd
from scipy import stats
from scipy.stats import norm

import math

class StatEssentials:
    @staticmethod
    def f_test_equal_variance(var1, var2, df1, df2):
        
        # Ensure var2 is the larger variance
        temp = var1 
        temp1 = df1
        if var1 < var2:
            var1 = var2 
            var2 = temp
            df1 = df2 
            df2 = temp1

        # Calculate the F-statistic
        f_statistic = var2 / var1

        # Calculate the p-value
        p_value = 1 - stats.f.cdf(f_statistic, df2, df1)
        return p_value

    @staticmethod
    def independent_two_sample_t_test_equal_variance(sample_mean1 , sample_mean2 , sample_variance1, sample_variance2, df1 , df2):
        pooled_variance = (df1*sample_variance1 + df2*sample_variance2)/(df1 + df2)
        test_statistic = (sample_mean1 - sample_mean2)/ math.sqrt(pooled_variance)
        cumulative_probability_right_tail = 1 - norm.cdf(abs(test_statistic))
        return 2 * cumulative_probability_right_tail
    @staticmethod
    def calculate_bin_number_histogram_series_obj(s):
        # Calculate the interquartile range
        iqr = s.quantile(0.75) - s.quantile(0.25)
        n = len(s)
        bin_width = 2 * iqr / (n ** (1/3))
        
        # Calculate the number of bins
        num_bins = int(np.ceil((s.max() - s.min()) / bin_width))
        return num_bins
    @staticmethod
    def conduct_shapiro_wilk_test(sample_panda_series):
        stat , p_value = stats.shapiro(sample_panda_series)
        if(p_value >0.05):
            return True
        else:
            return False



# # Example DataFrame
# data = {
#     'WindGustDir': ['N', 'E', 'S', 'N', 'E', 'S'],
#     'WindGustSpeed': [10, 15, 12, 11, 16, 14]
# }
# df = pd.DataFrame(data)

# # Group by 'WindGustDir'
# grouped = df.groupby('WindGustDir')

# # Iterate over groups
# for category, group in grouped:
#     print(f'Category: {category}')
#     print(group)