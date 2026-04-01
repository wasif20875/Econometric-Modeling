import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import gurobipy as gp
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
import matplotlib.dates as mdates
class FactorAnalysis:

    def __init__(self, stock_file: str, factors_file: str):
        self.stock_file = stock_file
        self.factors_file = factors_file
        self.data = pd.DataFrame()

    def prepare_data(self) -> None:
        # Load stock returns data
        stock_returns = pd.read_excel(self.stock_file, index_col="Date", parse_dates=True)
        print('stock_returns dataframe')
        print(stock_returns.head())

        # Load Fama-French factor data
        fama_french_factors = pd.read_csv(self.factors_file)
        fama_french_factors['Date'] = pd.to_datetime(fama_french_factors['Unnamed: 0'], format='%Y%m%d')
        fama_french_factors.set_index('Date', inplace=True)
        fama_french_factors.drop(['Unnamed: 0'], axis=1, inplace=True)
        print('fama_french dataframe')
        print(fama_french_factors.head())

        # Merge the datasets
        self.data = stock_returns.join(fama_french_factors, how="inner")
        print('merged both dataframes')

    def fit_model(self) -> None:
        X = self.data[["Mkt-RF", "SMB", "HML"]]
        y = self.data["Returns"] - self.data["RF"]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        print(model.summary())

    def fit_rolling_model(self, model_type: str, window_size: int) -> dict:
        if model_type == 'Rough':
            self.window_size = window_size
            rolling_params = []
            rolling_pvalues = []
            time_period = []

            X1 = self.data[["Mkt-RF", "SMB", "HML"]]
            y1 = self.data["Returns"] - self.data["RF"]
            start = 0

            for k in range(len(self.data) // self.window_size):
                y_rolling = y1.iloc[start:start + self.window_size]
                X_rolling = X1.iloc[start:start + self.window_size]
                time_period.append(self.data.index.date.tolist()[start])
                start += self.window_size

                X_rolling = sm.add_constant(X_rolling)
                model = sm.OLS(y_rolling, X_rolling).fit()
                rolling_pvalues.append(model.pvalues)
                rolling_params.append(model.params)

            # Handle remaining data
            y_rolling = y1.iloc[start:]
            X_rolling = X1.iloc[start:]
            time_period.append(self.data.index.date.tolist()[start])
            X_rolling = sm.add_constant(X_rolling)
            model = sm.OLS(y_rolling, X_rolling).fit()
            rolling_pvalues.append(model.pvalues)
            rolling_params.append(model.params)

            # Create DataFrames
            rolling_pvalues_df = pd.DataFrame(rolling_pvalues, columns=['const', 'Mkt-RF', 'SMB', 'HML'])
            rolling_pvalues_df['date'] = time_period

            rolling_params_df = pd.DataFrame(rolling_params, columns=['const', 'Mkt-RF', 'SMB', 'HML'])
            rolling_params_df['date'] = time_period

            # Plotting p-values
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in ['Mkt-RF', 'SMB', 'HML']:
                ax.plot(rolling_pvalues_df['date'], rolling_pvalues_df[col], label=f'Rolling P-Value - {col}')

            ax.set_xlabel('Date')
            ax.set_ylabel('P-Value')
            ax.set_title('Rolling P-Values for Fama-French Factors')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=90)
            plt.legend()

            # Plotting rolling beta coefficients
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for col in ['Mkt-RF', 'SMB', 'HML']:
                ax1.plot(rolling_params_df['date'], rolling_params_df[col], label=f'Rolling Beta - {col}')

            ax1.set_xlabel('Date')
            ax1.set_ylabel('Beta Value')
            ax1.set_title('Rolling Betas for Fama-French Factors')
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=90)
            plt.legend()

            return {'p_values': fig, 'params': fig1}

        else:
            self.window_size = window_size
            rolling_params = []
            rolling_pvalues = []
            X1 = self.data[["Mkt-RF", "SMB", "HML"]]  # Fama-French factors
            y1 = self.data["Returns"] - self.data["RF"]
            # time_period = []
            for start in range(len(self.data) - self.window_size + 1):
                # Rolling window data
                y_rolling = y1.iloc[start:start + self.window_size]
                X_rolling = X1.iloc[start:start + self.window_size]
                X_rolling = sm.add_constant(X_rolling)
                # Fit the OLS model
                model = sm.OLS(y_rolling, X_rolling).fit()
                rolling_pvalues.append(model.pvalues)
                rolling_params.append(model.params)
            rolling_pvalues_df = pd.DataFrame(rolling_pvalues, columns=['const', 'Mkt-RF', 'SMB', 'HML'])

            rolling_pvalues_df['date'] = self.data.index[:len(self.data) - window_size + 1].tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in ['Mkt-RF', 'SMB', 'HML']:
                ax.plot(rolling_pvalues_df['date'], rolling_pvalues_df[col], label=f'Rolling P-Value - {col}')

            ax.set_xlabel('Date')
            ax.set_ylabel('P-Value')
            ax.set_title('Rolling P-Values for Fama-French Factors')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=90)
            plt.legend() 

            rolling_params_df = pd.DataFrame(rolling_params, columns=['const', 'Mkt-RF', 'SMB', 'HML'])

            rolling_params_df['date'] = self.data.index[:len(self.data) - window_size + 1].tolist()

            # Plotting rolling beta coefficients
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for col in ['Mkt-RF', 'SMB', 'HML']:
                ax1.plot(rolling_params_df['date'], rolling_params_df[col], label=f'Rolling Beta - {col}')

            ax1.set_xlabel('Date')
            ax1.set_ylabel('Beta Value')
            ax1.set_title('Rolling Betas for Fama-French Factors')
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=90)
            plt.legend()

            return {'p_values': fig, 'params': fig1}
        
class TrackingErrorMinimization:

    def __init__(self, industry_file: str, portfolio_file: str):
        self.industry_file = industry_file
        self.portfolio_file = portfolio_file
        self.data = pd.DataFrame()
        self.industry_returns = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()

    def prepare_data(self) -> None:
        industry_returns = pd.read_csv(self.industry_file)
        portfolio_returns = pd.read_csv(self.portfolio_file)
        industry_returns['Date'] = pd.to_datetime(industry_returns['Unnamed: 0'], format='%Y%m')
        industry_returns.set_index('Date', inplace=True)
        industry_returns.drop(['Unnamed: 0'], inplace=True, axis=1)
        self.industry_returns = industry_returns
        print('Industry_returns data frame')
        print(industry_returns.head())

        portfolio_returns['Date'] = pd.to_datetime(portfolio_returns['Unnamed: 0'], format='%Y-%m')
        portfolio_returns.set_index('Date', inplace=True)
        portfolio_returns.drop(['Unnamed: 0'], inplace=True, axis=1)
        self.portfolio_returns = portfolio_returns
        print('Portfolio_returns data frame')
        print(portfolio_returns.head())

    def opt_weights_bargraph(self) -> dict:
        X = self.industry_returns
        X = sm.add_constant(X)
        y = self.portfolio_returns['Portfolio returns']

        def optimisation(data, y, X):
            num_industries = X.shape[1]
            num_periods = len(y)
            industry_returns_matrix = X.to_numpy()
            portfolio_returns_array = y.to_numpy()

            model = gp.Model("Tracking_Error_Minimization")
            model.Params.OutputFlag = 0
            weights = model.addVars(num_industries, lb=-1, ub=1, name="weights")
            tracking_error = gp.QuadExpr()

            for t in range(num_periods):
                expr = portfolio_returns_array[t] - gp.quicksum(weights[i] * industry_returns_matrix[t, i] for i in range(num_industries))
                tracking_error += expr * expr

            model.setObjective(tracking_error, gp.GRB.MINIMIZE)
            model.addConstr(gp.quicksum(weights[i] for i in range(num_industries)) == 1, "sum_weights")
            model.optimize()

            betas = [[X.columns[i], weights[i].X] for i in range(num_industries)]
            return betas

        betas = optimisation(self.industry_returns, y, X)

        labels, values = zip(*betas)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(labels, values, color='skyblue')

        for i, value in enumerate(values):
            ax.text(i, value + (0.01 if value >= 0 else -0.02), f'{value:.4f}', 
                    ha='center', va='bottom' if value >= 0 else 'top')

        ax.set_xlabel('Industry')
        ax.set_ylabel('Optimal Weight')
        ax.set_title('Optimal Weights for Tracking Error Minimization')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        return {'bar_graph':fig}
    
    def opt_weights_rolling(self, window: int) -> dict:
        X = self.industry_returns
        X = sm.add_constant(X)
        y = self.portfolio_returns['Portfolio returns']

        def rolling_optimisation(data, y, X, window):
            betas = pd.DataFrame(index=data.index[window:], columns=[
                "Alpha", "Beta_NoDur", "Beta_Durbl", "Beta_Manuf",
                "Beta_Enrgy", "Beta_HiTec", "Beta_Telcm", "Beta_Shops",
                "Beta_Hlth", "Beta_Utils", "Beta_Other"
            ])

            for i in range(window, len(data)):
                X_window = X.iloc[i - window:i]
                y_window = y.iloc[i - window:i]

                num_industries = X_window.shape[1]
                num_periods = len(y_window)

                industry_returns_matrix = X_window.to_numpy()
                portfolio_returns_array = y_window.to_numpy()

                model = gp.Model("Tracking_Error_Minimization")
                model.Params.OutputFlag = 0
                weights = model.addVars(num_industries, lb=-1, ub=1, name="weights")
                tracking_error = gp.QuadExpr()
                for t in range(num_periods):
                    expr = float(portfolio_returns_array[t]) - gp.quicksum(
                        weights[j] * float(industry_returns_matrix[t, j]) for j in range(num_industries)
                    )
                    tracking_error += expr * expr
                model.setObjective(tracking_error, gp.GRB.MINIMIZE)
                model.addConstr(gp.quicksum(weights[j] for j in range(num_industries)) == 1, "sum_weights")
                model.optimize()

                betas.iloc[i - window] = [weights[j].X for j in range(num_industries)]

            return betas

        betas = rolling_optimisation(self.industry_returns, y, X, window)

        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # Plotting betas with consistent styling
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'gray', 'black']
        labels = ["Beta_NoDur", "Beta_Durbl", "Beta_Manuf", "Beta_Enrgy", "Beta_HiTec", 
                "Beta_Telcm", "Beta_Shops", "Beta_Hlth", "Beta_Utils", "Beta_Other"]

        for label, color in zip(labels, colors):
            ax1.plot(betas.index, betas[label], label=label, color=color)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Rolling Regression Betas', fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Beta', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True)

        # Return the figure object instead of showing it
        return {'rolling_weights':fig1}
    
class CurrencyModel:

    def __init__(self, exchange_rates: str, bop_factors: str):
        self.exchange_rates = exchange_rates
        self.bop_factors = bop_factors
        self.data = pd.DataFrame()

    def prepare_data(self) -> None:
        exchange_rates = pd.read_excel(self.exchange_rates)
        bop_factors = pd.read_excel(self.bop_factors)
        exchange_rates.columns = ['Year', 'year_end', 'year_average']
        exchange_rates.index = exchange_rates.Year
        bop_factors.columns = [
            "Time Period / ACTUALS",
            "Current Account Deficit(CAD)/GDP",
            "Current Receipts/ Current Payments",
            "Current Receipts/ GDP",
            "Exports/ GDP",
            "Foreign Investment/ Exports",
            "Foreign Investment/ GDP",
            "Imports of Reserve",
            "Imports/ GDP",
            "Net/ GDP",
            "Payments/ GDP",
            "Receipts/ GDP"
        ]
        bop_factors.index = bop_factors['Time Period / ACTUALS']
        ex_bop_data = exchange_rates.join(bop_factors, how="inner")
        ex_bop_data['Foreign Investment/ Exports'] =ex_bop_data['Foreign Investment/ Exports'].interpolate(method = 'linear')
        ex_bop_data['Foreign Investment/ GDP'] =ex_bop_data['Foreign Investment/ GDP'].interpolate(method = 'linear')
        ex_bop_data.sort_index(inplace = True)
        self.data = ex_bop_data
        print('The data is created')
        print(ex_bop_data.info())
        print(ex_bop_data.head())

    def pairplot(self) -> None:
        sns.pairplot(self.data[list(set(self.data.columns.to_list()) - set(['Year','year_end','Time Period / ACTUALS']))])

    def drop(self, list1: list) -> None:
        self.data.drop(list1,axis = 1, inplace = True)
        print(f'Dropped columns {list1}')

    def vif(self) -> None:

        # Define target and features
        y = self.data['year_average']
        X = self.data.drop(columns=['year_average'])

        # Check for multicollinearity using VIF
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("Variance Inflation Factor (VIF):")
        print(vif_data)

    def heatmap(self) -> None:
        sns.heatmap(self.data.corr(),annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    def adfullertest(self) -> None:
        from statsmodels.tsa.stattools import adfuller
        self.data = self.data.sort_index()

        for i in (set(self.data.columns[:])):
            # Step 1: ADF Test on level data
            print('@@@@@@@@@@@',i)
            result_level = adfuller(self.data[i])
            print("ADF Test on Level Data:")
            print(f"ADF Statistic: {result_level[0]:.4f}")
            print(f"p-value: {result_level[1]:.4f}")
            
            if result_level[1] < 0.05:
                print("=> Series is stationary (reject null hypothesis of unit root).")
            else:
                print("=> Series is non-stationary. Testing first difference...")
            
                # Step 2: ADF Test on first difference
                self.data['diff'+i] = self.data[i].diff()
                result_diff = adfuller(self.data['diff'+i].dropna())
                print("\nADF Test on First Difference:")
                print(f"ADF Statistic: {result_diff[0]:.4f}")
                print(f"p-value: {result_diff[1]:.4f}")
            
                if result_diff[1] < 0.05:
                    print("=> First difference is stationary (series is I(1)).")
                else:
                    print("=> First difference is also non-stationary (series may be higher order).")

        self.data = self.data[self.data.columns[0:7].tolist()]

    
    def graph_trend(self) -> None:
        plt.figure(figsize=(12, 6))
        for column in self.data.columns[0:]:
            plt.plot(self.data.index.to_list(), self.data[column], marker='o', label=column)

        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Line Graph of Various Economic Indicators")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.show()

        ex_bop_data2 = self.data.diff().dropna()

        plt.figure(figsize=(12, 6))
        for column in ex_bop_data2.columns[0:]:
            plt.plot(ex_bop_data2.index.to_list(), ex_bop_data2[column], marker='o', label=column)

        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Line Graph of Various differenced Economic Indicators")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.show()

    def cointtest(self) -> None:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        # Prepare only the relevant columns for cointegration test
        # Johansen's test treats all variables symmetrically
        coint_data = self.data

        jres = coint_johansen(coint_data, det_order=0, k_ar_diff=1)

        print("Eigenvalues:\n", jres.eig)


        print("\nTrace Statistics:\n", jres.lr1)


        print("\nCritical Values for Trace Statistics:\n", jres.cvt)


        print("\nMax-Eigen Statistics:\n", jres.lr2)


        print("\nCritical Values for Max-Eigen:\n", jres.cvm)

    def lagcheck(self) -> None:
        from statsmodels.tsa.api import VAR

        # Define the endogenous variables
        endog_vars = self.data[['Current Account Deficit(CAD)/GDP', 'Net/ GDP', 'Imports of Reserve',
                                'Current Receipts/ Current Payments', 'Payments/ GDP', 'year_average',
                                'Foreign Investment/ Exports']]

        # Fit a VAR model to determine the optimal lag order
        var_model = VAR(endog_vars)
        lag_selection = var_model.select_order(maxlags=5)  # You can adjust maxlags as needed

        # Print the optimal lags for different criteria
        print("Optimal lags based on:")
        print(f"AIC: {lag_selection.aic}")
        print(f"BIC: {lag_selection.bic}")
        print(f"HQIC: {lag_selection.hqic}")
    
    def fitvecm(self) -> dict:
        self.data.set_index(pd.to_datetime(self.data['year_average'].index, format = '%Y'), inplace=True)
        from statsmodels.tsa.vector_ar.vecm import VECM

        vecm_model = VECM(self.data[['Current Account Deficit(CAD)/GDP', 'Net/ GDP', 'Imports of Reserve',
            'Current Receipts/ Current Payments', 'Payments/ GDP', 'year_average',
            'Foreign Investment/ Exports']][:], k_ar_diff=1, coint_rank=3)
        vecm_result = vecm_model.fit()
        print('k_ar_diff is taken as 1 and no of cointegratino relationships are 3')
        
        return {'data':self.data,'model':vecm_result}

    def fit_vs_actual(self,model) -> plt.Figure:
        self.vecm_result = model
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract the fitted values from the VECM model
        y_fitted = [float(self.vecm_result.fittedvalues[i][5]) for i in range(self.vecm_result.fittedvalues.shape[0])]

        # Extract the actual values and years
        years = list(self.data.index[self.data.shape[0]-len(y_fitted):])  # Adjusted to match the fitted values length
        actual_values = self.data['year_average'][self.data.shape[0]-len(y_fitted):]

        # Convert to numpy arrays for calculation
        y_fitted = np.array(y_fitted)
        actual_values = np.array(actual_values)

        # Plot the actual vs fitted values
        plt.figure(figsize=(12, 6))
        plt.plot(years, actual_values, label='Actual Exchange Rate', color='blue', marker='o')
        plt.plot(years, y_fitted, label='Fitted Values', color='red', linestyle='--', marker='x')

        # Adding labels and title
        plt.title('Actual vs Fitted Exchange Rate Values with Standard Deviation Bands')
        plt.xlabel('Year')
        plt.ylabel('Exchange Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

    def vecm_tests(self,model) -> None:
        self.vecm_result = model
        #Running Residual tests
        y_fitted = [float(self.vecm_result.fittedvalues[i][5]) for i in range(self.vecm_result.fittedvalues.shape[0])]
        fitted_values = y_fitted
        actual_values = self.data['year_average'][self.data.shape[0]-len(y_fitted):]
        # Model Evaluation
        # print("R-squared:", r2_score(actual_values, fitted_values))
        # print("RMSE:", np.sqrt(mean_squared_error(actual_values, fitted_values)))

        # Check residuals
        residuals = actual_values - fitted_values
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")
        plt.show()

        # Normality test (Shapiro-Wilk Test)
        shapiro_test = shapiro(residuals)
        print("Shapiro-Wilk Test p-value:", shapiro_test.pvalue)

        # Plot actual vs predicted
        plt.scatter(actual_values, fitted_values)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.show()

        # Check for homoscedasticity
        plt.scatter(fitted_values, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Homoscedasticity Check")
        plt.show()

        # # Model Summary using Statsmodels
        # X_train_const = sm.add_constant(X_train)
        # ols_model = sm.OLS(y_train, X_train_const).fit()
        # print(ols_model.summary())


        serial_test = self.vecm_result.test_whiteness()
        print("Serial Correlation Test:\n", serial_test)

        # Normality Test (Jarque-Bera)
        normality_test = self.vecm_result.test_normality()
        print("Normality Test:\n", normality_test)

        # Heteroskedasticity Test (White's Test)
        heteroskedasticity_test = self.vecm_result.test_whiteness()
        print("Heteroskedasticity Test:\n", heteroskedasticity_test)