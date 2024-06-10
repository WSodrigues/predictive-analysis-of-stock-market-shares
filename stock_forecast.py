import logging
import os
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import optuna
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from imblearn.over_sampling import SMOTE
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import shap

# Load environment variables from .env file
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
WEATHERAPI_API_KEY = os.getenv('WEATHERAPI_API_KEY')
WEATHERBIT_API_KEY = os.getenv('WEATHERBIT_API_KEY')
EVENTREGISTRY_API_KEY = os.getenv('EVENTREGISTRY_API_KEY')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# File paths
TICKERS_FILE = 'tikers.txt'
ERRORS_FILE = 'errors.txt'
UPDATE_INTERVAL_DAYS = 7
LAST_UPDATE_FILE = 'last_update.txt'


def exibir_aviso_de_risco():
    print(
        "AVISO: O desempenho passado não garante resultados futuros. "
        "O mercado de ações envolve riscos substanciais. Consulte um conselheiro financeiro antes de tomar decisões de investimento."
    )


def baixar_dados(ticker, start_date, end_date):
    try:
        logging.info(f"Baixando dados para {ticker} de {start_date} até {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.isnull().values.any():
            logging.warning("Dados incompletos foram encontrados e removidos.")
            df.dropna(inplace=True)
        df = calcular_indicadores(df)
        logging.info(f"Dados para {ticker} baixados e indicadores calculados com sucesso.")
        return df
    except Exception as e:
        logging.error(f"Erro ao baixar dados para {ticker}: {e}")
        return None


def calcular_indicadores(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    df['BB_Upper'], df['BB_Lower'] = compute_BB(df['Close'])
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['MFI'] = compute_MFI(df)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = compute_ADX(df)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['LogRet'].rolling(window=10).std()
    df['CCI'] = compute_CCI(df)
    df['ATR'] = compute_ATR(df)
    df['Stochastic'] = compute_stochastic(df)
    df.dropna(inplace=True)

    df['VIX'] = yf.download('^VIX', start=df.index[0], end=df.index[-1])['Close']
    df['UST10Y'] = yf.download('^TNX', start=df.index[0], end=df.index[-1])['Close']
    df['USD_Index'] = yf.download('DX-Y.NYB', start=df.index[0], end=df.index[-1])['Close']
    return df


def compute_RSI(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


def compute_MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    exp1 = data.ewm(span=fastperiod, adjust=False).mean()
    exp2 = data.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal


def compute_BB(data, window=20, num_std=2):
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def compute_MFI(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
    money_flow_ratio = positive_flow / negative_flow
    MFI = 100 - (100 / (1 + money_flow_ratio))
    return MFI


def compute_ADX(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = np.maximum((high - low), np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()
    return adx


def compute_CCI(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cci = (tp - tp.rolling(window=window).mean()) / (0.015 * tp.rolling(window=window).std())
    return cci


def compute_ATR(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def compute_stochastic(df, window=14):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stoch = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return stoch


def analyze_sentiment(ticker, date, api_tool, api_key=None, use_social_searcher=False):
    global response
    if api_tool == "newsapi" and api_key:
        url = f"https://newsapi.org/v2/everything?q={ticker}&from={date}&sortBy=popularity&apiKey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if 'articles' not in data:
                logging.error(f"The response does not contain 'articles'. Full response: {data}")
                return 0

            sentiments = [np.random.choice(['positive', 'negative', 'neutral']) for _ in data['articles']]
            return sum(1 for s in sentiments if s == 'positive') / len(sentiments)
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 426:
                logging.error("News API key requires an upgrade.")
            else:
                logging.error(f"HTTP error occurred: {http_err}")
            return 0
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling News API: {e}")
            return 0
    elif use_social_searcher:
        url = f"https://www.social-searcher.com/{ticker}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            sentiment_data = soup.find('div', {'class': 'item sentiment'})
            if sentiment_data:
                positive = sentiment_data.find('span', {'class': 'positive'})
                negative = sentiment_data.find('span', {'class': 'negative'})
                if positive and negative:
                    positive_value = int(positive.text.strip('%'))
                    negative_value = int(negative.text.strip('%'))
                    sentiment_score = positive_value / (positive_value + negative_value)
                    return sentiment_score
            return 0
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao chamar a Social Mention: {e}")
            return 0
    else:
        logging.info("Nenhuma opção de análise de sentimento foi fornecida.")
        return 0


def etiquetar_dados_dinamicamente(df, vix_value, sentimento):
    df['Etiqueta'] = 'Manter'
    if vix_value > 25:
        df.loc[(df['RSI'] > 70) & (df['MFI'] > 80), 'Etiqueta'] = 'Venda'
        df.loc[(df['RSI'] < 30) & (df['MFI'] < 20), 'Etiqueta'] = 'Compra'
    else:
        df.loc[(df['RSI'] > 65) & (df['MFI'] > 75), 'Etiqueta'] = 'Venda'
        df.loc[(df['RSI'] < 35) & (df['MFI'] < 25), 'Etiqueta'] = 'Compra'

    if sentimento > 0.6:
        df.loc[(df['Etiqueta'] == 'Manter') & (df['RSI'] < 50), 'Etiqueta'] = 'Compra'
    elif sentimento < 0.4:
        df.loc[(df['Etiqueta'] == 'Manter') & (df['RSI'] > 50), 'Etiqueta'] = 'Venda'

    if 'Compra' not in df['Etiqueta'].values:
        df = pd.concat([df, pd.DataFrame({'Etiqueta': ['Compra']})], ignore_index=True)
    if 'Venda' not in df['Etiqueta'].values:
        df = pd.concat([df, pd.DataFrame({'Etiqueta': ['Venda']})], ignore_index=True)
    if 'Manter' not in df['Etiqueta'].values:
        df = pd.concat([df, pd.DataFrame({'Etiqueta': ['Manter']})], ignore_index=True)

    return df


def optimize_ensemble_with_optuna(X_train_smote, y_train_smote, X_valid, y_valid, X_test, y_test):
    def objective(trial):
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
        rf_max_depth = trial.suggest_int('rf_max_depth', 3, 50)
        rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
        rf_min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 20)
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 50)
        xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
        lgbm_n_estimators = trial.suggest_int('lgbm_n_estimators', 50, 300)
        lgbm_max_depth = trial.suggest_int('lgbm_max_depth', 3, 50)
        lgbm_num_leaves = trial.suggest_int('lgbm_num_leaves', 31, 256)
        lgbm_learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.3)

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf, random_state=42
        )
        xgb = XGBClassifier(
            n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate,
            random_state=42, eval_metric='mlogloss'
        )
        lgbm = LGBMClassifier(
            n_estimators=lgbm_n_estimators, max_depth=lgbm_max_depth, num_leaves=lgbm_num_leaves,
            learning_rate=lgbm_learning_rate, random_state=42
        )

        rf.fit(X_train_smote, y_train_smote)
        xgb.fit(
            X_train_smote, y_train_smote, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10,
            verbose=False
        )
        lgbm.fit(
            X_train_smote, y_train_smote, eval_set=[(X_valid, y_valid)], eval_metric='logloss',
            callbacks=[early_stopping(10)]
        )

        ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)], voting='soft')
        ensemble.fit(X_train_smote, y_train_smote)
        y_pred = ensemble.predict(X_valid)

        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    logging.info(f"Melhores parâmetros obtidos: {best_params}")

    rf = RandomForestClassifier(
        n_estimators=best_params['rf_n_estimators'], max_depth=best_params['rf_max_depth'],
        min_samples_split=best_params['rf_min_samples_split'],
        min_samples_leaf=best_params['rf_min_samples_leaf'],
        random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=best_params['xgb_n_estimators'], max_depth=best_params['xgb_max_depth'],
        learning_rate=best_params['xgb_learning_rate'], random_state=42, eval_metric='mlogloss'
    )
    lgbm = LGBMClassifier(
        n_estimators=best_params['lgbm_n_estimators'], max_depth=best_params['lgbm_max_depth'],
        num_leaves=best_params['lgbm_num_leaves'], learning_rate=best_params['lgbm_learning_rate'],
        random_state=42
    )

    rf.fit(X_train_smote, y_train_smote)
    xgb.fit(
        X_train_smote, y_train_smote, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10,
        verbose=False
    )
    lgbm.fit(
        X_train_smote, y_train_smote, eval_set=[(X_valid, y_valid)], eval_metric='logloss',
        callbacks=[early_stopping(10)]
    )

    best_ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)], voting='soft'
    )
    best_ensemble.fit(X_train_smote, y_train_smote)
    y_pred = best_ensemble.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(
            label_binarize(y_test, classes=[0, 1, 2]),
            best_ensemble.predict_proba(X_test), multi_class='ovr'
        )
    except ValueError as e:
        logging.warning(f"ROC AUC score could not be calculated: {e}")
        roc_auc = None

    return best_ensemble, accuracy, precision, recall, f1, roc_auc


def sugerir_alocacao(ticker, probabilidades, tolerancia_risco, volatilidade):
    prob_compra = probabilidades[1]
    prob_venda = probabilidades[2]
    prob_manter = probabilidades[0]
    sugestoes = {
        'comprar': "Não recomendado conforme a sua escolha de tolerância ao risco.",
        'vender': "Não recomendado conforme a sua escolha de tolerância ao risco.",
        'manter': "Recomendado conforme a sua escolha de tolerância ao risco."
    }

    if tolerancia_risco == 'Alta':
        if prob_compra > 0.7:
            sugestoes['comprar'] = f"Alocar 20% do portfólio para {ticker}, Stop Loss: 85%, Take Profit: 150%"
        elif prob_compra > 0.5:
            sugestoes['comprar'] = f"Alocar 10% do portfólio para {ticker}, Stop Loss: 90%, Take Profit: 140%"
        if prob_venda > 0.7:
            sugestoes['vender'] = f"Reduzir 20% do portfólio em {ticker}, Stop Loss: 105%, Take Profit: 70%"
        elif prob_venda > 0.5:
            sugestoes['vender'] = f"Reduzir 10% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 75%"

    elif tolerancia_risco == 'Média':
        if prob_compra > 0.6:
            sugestoes['comprar'] = f"Alocar 15% do portfólio para {ticker}, Stop Loss: 90%, Take Profit: 130%"
        elif prob_compra > 0.4:
            sugestoes['comprar'] = f"Alocar 7% do portfólio para {ticker}, Stop Loss: 95%, Take Profit: 120%"
        if prob_venda > 0.6:
            sugestoes['vender'] = f"Reduzir 15% do portfólio em {ticker}, Stop Loss: 102%, Take Profit: 75%"
        elif prob_venda > 0.4:
            sugestoes['vender'] = f"Reduzir 7% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 80%"

    elif tolerancia_risco == 'Baixa':
        if prob_compra > 0.5:
            sugestoes['comprar'] = f"Alocar 10% do portfólio para {ticker}, Stop Loss: 95%, Take Profit: 110%"
        elif prob_compra > 0.3:
            sugestoes['comprar'] = f"Alocar 5% do portfólio para {ticker}, Stop Loss: 98%, Take Profit: 105%"
        if prob_venda > 0.5:
            sugestoes['vender'] = f"Reduzir 10% do portfólio em {ticker}, Stop Loss: 101%, Take Profit: 80%"
        elif prob_venda > 0.3:
            sugestoes['vender'] = f"Reduzir 5% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 85%"

    if volatilidade > 0.02:
        sugestoes[
            'manter'] = f"Volatilidade alta detectada, sugere-se manter {ticker} e evitar novas posições até que a volatilidade diminua."

    return sugestoes


@contextmanager
def temporary_files_cleanup():
    temp_files = []
    yield temp_files
    for file in temp_files:
        try:
            os.remove(file)
        except OSError as e:
            logging.warning(f"Failed to remove temporary file {file}: {e}")


def fetch_weather_data(location, api_key, provider):
    if provider == 'openweathermap':
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    elif provider == 'weatherapi':
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    elif provider == 'weatherbit':
        url = f"http://api.weatherbit.io/v2.0/current?city={location}&key={api_key}"
    else:
        logging.error("Invalid weather provider.")
        return None

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        return None


def fetch_event_data(api_key):
    url = f"https://eventregistry.org/api/v1/article/getArticles?apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching event data: {e}")
        return None


def processar_acao(ticker, start_date, end_date, tolerancia_risco, api_tool, api_key, use_social_searcher):
    global X_train, X_test, y_train, y_test
    try:
        dados_acao = baixar_dados(ticker, start_date, end_date)
        if dados_acao is None:
            return None, None, None, False

        vix_value = dados_acao['VIX'].iloc[-1]
        sentimento = analyze_sentiment(ticker, dados_acao.index[-30], api_tool, api_key, use_social_searcher)
        use_sentiment = api_key or use_social_searcher

        # Fetch weather data
        weather_data = fetch_weather_data("New York", OPENWEATHER_API_KEY, 'openweathermap')
        if weather_data:
            logging.info(f"Weather data for New York: {weather_data}")

        # Fetch event data
        event_data = fetch_event_data(EVENTREGISTRY_API_KEY)
        if event_data:
            logging.info(f"Event data: {event_data}")

        dados_acao = etiquetar_dados_dinamicamente(dados_acao, vix_value, sentimento)

        logging.info(f"Distribuição das classes para {ticker}:\n{dados_acao['Etiqueta'].value_counts()}")

        features = [
            'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'MA10', 'MA100',
            'EMA20', 'Momentum', 'MFI', 'EMA50', 'ADX', 'LogRet', 'Volatility', 'CCI', 'ATR',
            'Stochastic', 'VIX', 'UST10Y', 'USD_Index'
        ]

        # Add additional features from weather and event data
        if weather_data:
            features.extend(['Temp', 'Humidity', 'WeatherDesc'])
            dados_acao['Temp'] = weather_data['main']['temp']
            dados_acao['Humidity'] = weather_data['main']['humidity']
            dados_acao['WeatherDesc'] = weather_data['weather'][0]['description']

        if event_data:
            features.extend(['EventCount', 'EventSentiment'])
            dados_acao['EventCount'] = len(event_data['articles'])
            dados_acao['EventSentiment'] = sum(
                1 for s in event_data['articles'] if s['sentiment'] == 'positive'
            ) / len(event_data['articles'])

        X = dados_acao[features]
        y = dados_acao['Etiqueta']

        label_mapping = {'Compra': 0, 'Manter': 1, 'Venda': 2}
        y = y.map(label_mapping)

        # Create a pipeline with imputer, scaler and model
        pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', None)  # Placeholder for the model
            ]
        )

        present_classes = set(y)
        expected_classes = set(label_mapping.values())

        if not present_classes.issuperset(expected_classes):
            missing_classes = expected_classes - present_classes
            logging.warning(
                f"Classes {missing_classes} are not present in the data for {ticker}. Model performance may be affected."
            )

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if len(set(y_train)) == len(present_classes) and len(set(y_test)) == len(
                    present_classes):
                break

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            logging.error("Dados de treino ou teste contêm NaNs. Lidando com valores ausentes.")
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

        class_counts = y_train.value_counts()
        minority_class = class_counts.idxmin()
        n_minority_samples = class_counts.min()

        if n_minority_samples < 2:
            logging.warning(f"Not enough samples for minority class {minority_class} to apply SMOTE.")
            X_train_smote, y_train_smote = X_train, y_train
        else:
            smote_k = min(5, n_minority_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=smote_k)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        if n_minority_samples < 2:
            X_train_smote, X_valid, y_train_smote, y_valid = train_test_split(
                X_train_smote, y_train_smote, test_size=0.2, random_state=42
            )
        else:
            X_train_smote, X_valid, y_train_smote, y_valid = train_test_split(
                X_train_smote, y_train_smote, test_size=0.2, stratify=y_train_smote, random_state=42
            )

        best_model, accuracy, precision, recall, f1, roc_auc = optimize_ensemble_with_optuna(
            X_train_smote, y_train_smote, X_valid, y_valid, X_test, y_test
        )

        logging.info(f"Melhor precisão obtida: {accuracy}")

        explainer = shap.TreeExplainer(best_model.named_estimators_['xgb'])
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        shap_plot_path = "shap_summary_plot.png"
        plt.savefig(shap_plot_path)
        logging.info(f"SHAP summary plot saved at {shap_plot_path}")
        plt.close()

        dados_acao = dados_acao.asfreq('D')

        arima_model = ARIMA(dados_acao['Close'], order=(1, 1, 1))
        arima_results = arima_model.fit()
        arima_forecast = arima_results.forecast(steps=30)

        if len(arima_forecast) > 0:
            logging.info(f"ARIMA Previsão (30 dias): {arima_forecast.iloc[-1]}")
        else:
            logging.warning("ARIMA não conseguiu prever.")

        with temporary_files_cleanup() as temp_files:
            prophet_model = Prophet(daily_seasonality=True)
            prophet_df = pd.DataFrame({'ds': dados_acao.index, 'y': dados_acao['Close']})
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=30)
            prophet_forecast = prophet_model.predict(future)

            temp_file_path = tempfile.mkstemp()[1]
            temp_files.append(temp_file_path)
            with open(temp_file_path, 'w') as f:
                f.write(prophet_forecast.to_json())

        if not prophet_forecast.empty:
            logging.info(f"Prophet Previsão (30 dias): {prophet_forecast['yhat'].iloc[-1]}")
        else:
            logging.warning("Prophet não conseguiu prever.")

        X_recent = X[-30:]
        probabilities = best_model.predict_proba(X_recent)
        avg_probabilities = np.mean(probabilities, axis=0)
        sugestoes = sugerir_alocacao(
            ticker, avg_probabilities, tolerancia_risco, dados_acao['Volatility'].iloc[-30:].mean()
        )

        return best_model, accuracy, sugestoes, use_sentiment

    except Exception as e:
        logging.error(f"Erro ao processar ação {ticker}: {e}")
        return None, None, None, True  # Added the error flag


def validar_tickers(tickers):
    valid_tickers = []
    error_tickers = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='1d')
            if not df.empty:
                valid_tickers.append(ticker)
            else:
                logging.warning(f"Ticker {ticker} is invalid or delisted.")
                error_tickers.append(ticker)
        except Exception as e:
            logging.error(f"Error validating ticker {ticker}: {e}")
            error_tickers.append(ticker)
    return valid_tickers, error_tickers


def update_tickers():
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={ALPHA_VANTAGE_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        new_tickers = [item['symbol'] for item in data]
        with open(TICKERS_FILE, 'w') as file:
            file.write("\n".join(new_tickers))
        logging.info(f"Updated {TICKERS_FILE} with new tickers.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error updating tickers: {e}")


def check_and_update_tickers():
    if not os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, 'w') as file:
            file.write(str(datetime.now() - timedelta(days=UPDATE_INTERVAL_DAYS + 1)))

    with open(LAST_UPDATE_FILE, 'r') as file:
        last_update = datetime.fromisoformat(file.read().strip())

    if datetime.now() - last_update > timedelta(days=UPDATE_INTERVAL_DAYS):
        update_tickers()
        with open(LAST_UPDATE_FILE, 'w') as file:
            file.write(str(datetime.now()))


def main():
    global sugestoes
    check_and_update_tickers()  # Check and update tickers if necessary

    config_path = 'config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    start_date = '2000-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    tolerancia_risco = 'Alta'
    total_portfolio = 100000.0
    selected_currency = "USD"

    try:
        with open(TICKERS_FILE, 'r') as file:
            adicionar_tickers = file.read().splitlines()
    except FileNotFoundError:
        logging.error(f"{TICKERS_FILE} file not found. Please ensure the file exists in the project directory.")
        return

    api_tool = 'newsapi'
    api_key = NEWS_API_KEY
    use_social_searcher = False

    error_tickers = []
    if os.path.exists(ERRORS_FILE):
        with open(ERRORS_FILE, 'r') as file:
            error_tickers = file.read().splitlines()

    adicionar_tickers = [ticker for ticker in adicionar_tickers if ticker not in error_tickers]
    valid_tickers, new_error_tickers = validar_tickers(
        adicionar_tickers
    )  # Get the lists of valid and new error tickers

    # Combine old and new error tickers
    error_tickers = list(set(error_tickers + new_error_tickers))

    modelos = {}
    all_sugestoes = {}

    for ticker in valid_tickers:
        modelo, acuracia, sugestoes, error_flag = processar_acao(
            ticker, start_date, end_date, tolerancia_risco, api_tool, api_key, use_social_searcher
        )
        if modelo and sugestoes and not error_flag:
            modelos[ticker] = modelo
            all_sugestoes[ticker] = sugestoes
            acuracia_percent = round(acuracia * 100, 2)
            title = f"{ticker} (Acurácia: {acuracia_percent}%)"
            print(f"### {title}")
            print(f"**Comprar:**\n- {sugestoes['comprar']}")
            print(f"**Vender:**\n- {sugestoes['vender']}")
            print(f"**Manter:**\n- {sugestoes['manter']}")

            current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[0]
            print(f"Preço Atual de {ticker}: {selected_currency} {current_price:.2f}")

            conversion_rate = fetch_conversion_rate('USD', selected_currency)
            if conversion_rate is None:
                print(f"Falha ao buscar taxa de conversão para {selected_currency}.")
                continue

            print(f"## Cálculo para {ticker}")
            print(f"### Alocação de Compra")
            if "não recomendado" not in sugestoes['comprar'].lower():
                compra_percent = float(sugestoes['comprar'].split(' ')[1].strip('%'))
                alocar_compra_usd = (compra_percent / 100) * total_portfolio
                alocar_compra = convert_currency(alocar_compra_usd, conversion_rate)
                print(f"Alocar: {selected_currency} {alocar_compra:.2f}")
                print("**Fórmula:** Alocar = (Percentual de Compra / 100) * Total do Portfólio")
                print(
                    f"**Cálculo:** ({compra_percent:.2f} / 100) * {total_portfolio:.2f} = {alocar_compra_usd:.2f} USD = {alocar_compra:.2f} {selected_currency}"
                )

                quantidade_acoes = alocar_compra / current_price
                print(f"Quantidade de Ações: {quantidade_acoes:.2f}")
                stop_loss_compra = current_price * 0.98
                take_profit_compra = current_price * 1.05
                print(f"Stop Loss: {selected_currency} {stop_loss_compra:.2f}")
                print(f"Take Profit: {selected_currency} {take_profit_compra:.2f}")

            print(f"### Reduzir Alocação")
            if "não recomendado" not in sugestoes['vender'].lower():
                venda_percent = float(sugestoes['vender'].split(' ')[1].strip('%'))
                alocar_venda_usd = (venda_percent / 100) * total_portfolio
                alocar_venda = convert_currency(alocar_venda_usd, conversion_rate)
                print(f"Alocar: {selected_currency} {alocar_venda:.2f}")
                print("**Fórmula:** Alocar = (Percentual de Venda / 100) * Total do Portfólio")
                print(
                    f"**Cálculo:** ({venda_percent:.2f} / 100) * {total_portfolio:.2f} = {alocar_venda_usd:.2f} USD = {alocar_venda:.2f} {selected_currency}"
                )

                quantidade_venda = alocar_venda / current_price
                print(f"Quantidade de Ações: {quantidade_venda:.2f}")
                stop_loss_venda = current_price * 1.01
                take_profit_venda = current_price * 0.80
                print(f"Stop Loss: {selected_currency} {stop_loss_venda:.2f}")
                print(f"Take Profit: {selected_currency} {take_profit_venda:.2f}")

    best_buy, best_sell, best_hold = evaluate_best_options(all_sugestoes)
    print("\n### Best Recommendations Across All Tickers (Updated) ###")
    print(f"**Best Buy:**\n- {best_buy}")
    print(f"**Best Sell:**\n- {best_sell}")
    print(f"**Best Hold:**\n- {best_hold}")

    with open("conselhos.txt", "w") as file:
        for ticker in valid_tickers:
            if ticker in modelos:
                conselho = f"{ticker}: Comprar: {sugestoes['comprar']} Vender: {sugestoes['vender']} Manter: {sugestoes['manter']}"
                file.write(conselho + "\n")

    with open(ERRORS_FILE, "w") as file:
        for ticker in error_tickers:
            file.write(ticker + "\n")

    os.system("notepad.exe conselhos.txt")
    os.system("notepad.exe errors.txt")


def evaluate_best_options(all_sugestoes):
    best_buy = None
    best_sell = None
    best_hold = None

    max_buy_percent = 0
    max_sell_percent = 0
    max_hold_percent = 0

    for ticker, sugestoes in all_sugestoes.items():
        if "não recomendado" not in sugestoes['comprar'].lower():
            compra_percent = float(sugestoes['comprar'].split(' ')[1].strip('%'))
            if compra_percent > max_buy_percent:
                max_buy_percent = compra_percent
                best_buy = f"{ticker}: {sugestoes['comprar']}"

        if "não recomendado" not in sugestoes['vender'].lower():
            venda_percent = float(sugestoes['vender'].split(' ')[1].strip('%'))
            if venda_percent > max_sell_percent:
                max_sell_percent = venda_percent
                best_sell = f"{ticker}: {sugestoes['vender']}"

        if "não recomendado" not in sugestoes['manter'].lower():
            manter_percent = float(sugestoes['manter'].split(' ')[1].strip('%')) if '%' in sugestoes[
                'manter'] else 0
            if manter_percent > max_hold_percent:
                best_hold = f"{ticker}: {sugestoes['manter']}"

    return best_buy, best_sell, best_hold


def fetch_conversion_rate(from_currency, to_currency):
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        rates = data['rates']
        if to_currency in rates:
            return rates[to_currency]
        else:
            logging.error(f"Currency {to_currency} not found in the exchange rates.")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch exchange rate: {e}")
        return None


def convert_currency(amount, rate):
    return amount * rate


if __name__ == "__main__":
    exibir_aviso_de_risco()
    main()
