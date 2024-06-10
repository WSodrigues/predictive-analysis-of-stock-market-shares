import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import json
from sklearn.model_selection import train_test_split , TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , roc_auc_score
from sklearn.preprocessing import StandardScaler , label_binarize
from imblearn.over_sampling import SMOTE
import optuna
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier , early_stopping
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import requests
import tempfile
from contextlib import contextmanager
from bs4 import BeautifulSoup

# Configuração do logging
logging.basicConfig (
	level = logging.INFO ,
	format = '%(asctime)s - %(levelname)s - %(message)s'
	)


# Email alert function
def send_email_alert ( subject , body ) :
	sender_email = "your_email@example.com"
	receiver_email = "recipient_email@example.com"
	password = os.getenv ( 'EMAIL_PASSWORD' )
	
	message = MIMEMultipart ( )
	message [ "From" ] = sender_email
	message [ "To" ] = receiver_email
	message [ "Subject" ] = subject
	message.attach ( MIMEText ( body , "plain" ) )
	
	try :
		server = smtplib.SMTP_SSL ( "smtp.example.com" , 465 )
		server.login ( sender_email , password )
		server.sendmail ( sender_email , receiver_email , message.as_string ( ) )
		server.close ( )
		logging.info ( "Email alert sent successfully" )
	except Exception as e :
		logging.error ( f"Error sending email: {e}" )


# Ticker validation function
def validar_tickers ( tickers ) :
	valid_tickers = [ ]
	for ticker in tickers :
		try :
			df = yf.download ( ticker , period = '1d' )
			if not df.empty :
				valid_tickers.append ( ticker )
			else :
				error_message = f"Ticker {ticker} is invalid or delisted."
				logging.warning ( error_message )
				st.warning ( error_message )
		except Exception as e :
			error_message = f"Error validating ticker {ticker}: {e}"
			logging.error ( error_message )
			st.error ( error_message )
	return valid_tickers


# Load configuration function
@st.cache_data
def load_config ( config_path ) :
	with open ( config_path , 'r' ) as file :
		config = json.load ( file )
	return config


# Technical indicators calculation functions
def compute_RSI ( data , window = 14 ) :
	delta = data.diff ( 1 )
	gain = (delta.where ( delta > 0 , 0 )).rolling ( window = window ).mean ( )
	loss = (-delta.where ( delta < 0 , 0 )).rolling ( window = window ).mean ( )
	RS = gain / loss
	RSI = 100 - (100 / (1 + RS))
	return RSI


def compute_MACD ( data , fastperiod = 12 , slowperiod = 26 , signalperiod = 9 ) :
	exp1 = data.ewm ( span = fastperiod , adjust = False ).mean ( )
	exp2 = data.ewm ( span = slowperiod , adjust = False ).mean ( )
	macd = exp1 - exp2
	signal = macd.ewm ( span = signalperiod , adjust = False ).mean ( )
	return macd , signal


def compute_BB ( data , window = 20 , num_std = 2 ) :
	rolling_mean = data.rolling ( window ).mean ( )
	rolling_std = data.rolling ( window ).std ( )
	upper_band = rolling_mean + (rolling_std * num_std)
	lower_band = rolling_mean - (rolling_std * num_std)
	return upper_band , lower_band


def compute_MFI ( df , window = 14 ) :
	high = df [ 'High' ]
	low = df [ 'Low' ]
	close = df [ 'Close' ]
	volume = df [ 'Volume' ]
	typical_price = (high + low + close) / 3
	money_flow = typical_price * volume
	positive_flow = money_flow.where ( typical_price > typical_price.shift ( 1 ) , 0 ).rolling (
		window = window
		).sum ( )
	negative_flow = money_flow.where ( typical_price < typical_price.shift ( 1 ) , 0 ).rolling (
		window = window
		).sum ( )
	money_flow_ratio = positive_flow / negative_flow
	MFI = 100 - (100 / (1 + money_flow_ratio))
	return MFI


def compute_ADX ( df , window = 14 ) :
	high = df [ 'High' ]
	low = df [ 'Low' ]
	close = df [ 'Close' ]
	plus_dm = high.diff ( )
	minus_dm = low.diff ( )
	tr = np.maximum ( (high - low) , np.maximum ( abs ( high - close.shift ( ) ) , abs ( low - close.shift ( ) ) ) )
	plus_dm = plus_dm.where ( (plus_dm > minus_dm) & (plus_dm > 0) , 0.0 )
	minus_dm = minus_dm.where ( (minus_dm > plus_dm) & (minus_dm > 0) , 0.0 )
	atr = tr.rolling ( window = window ).mean ( )
	plus_di = 100 * (plus_dm.rolling ( window = window ).mean ( ) / atr)
	minus_di = 100 * (minus_dm.rolling ( window = window ).mean ( ) / atr)
	dx = 100 * (abs ( plus_di - minus_di ) / (plus_di + minus_di))
	adx = dx.rolling ( window = window ).mean ( )
	return adx


def compute_CCI ( df , window = 20 ) :
	tp = (df [ 'High' ] + df [ 'Low' ] + df [ 'Close' ]) / 3
	cci = (tp - tp.rolling ( window = window ).mean ( )) / (0.015 * tp.rolling ( window = window ).std ( ))
	return cci


def compute_ATR ( df , window = 14 ) :
	high_low = df [ 'High' ] - df [ 'Low' ]
	high_close = np.abs ( df [ 'High' ] - df [ 'Close' ].shift ( ) )
	low_close = np.abs ( df [ 'Low' ] - df [ 'Close' ].shift ( ) )
	ranges = pd.concat ( [ high_low , high_close , low_close ] , axis = 1 )
	true_range = np.max ( ranges , axis = 1 )
	atr = true_range.rolling ( window = window ).mean ( )
	return atr


def compute_stochastic ( df , window = 14 ) :
	low_min = df [ 'Low' ].rolling ( window = window ).min ( )
	high_max = df [ 'High' ].rolling ( window = window ).max ( )
	stoch = 100 * ((df [ 'Close' ] - low_min) / (high_max - low_min))
	return stoch


# Calculate all indicators
def calcular_indicadores ( df ) :
	df [ 'MA20' ] = df [ 'Close' ].rolling ( window = 20 ).mean ( )
	df [ 'MA50' ] = df [ 'Close' ].rolling ( window = 50 ).mean ( )
	df [ 'RSI' ] = compute_RSI ( df [ 'Close' ] )
	df [ 'MACD' ] , df [ 'MACD_Signal' ] = compute_MACD ( df [ 'Close' ] )
	df [ 'BB_Upper' ] , df [ 'BB_Lower' ] = compute_BB ( df [ 'Close' ] )
	df [ 'MA10' ] = df [ 'Close' ].rolling ( window = 10 ).mean ( )
	df [ 'MA100' ] = df [ 'Close' ].rolling ( window = 100 ).mean ( )
	df [ 'EMA20' ] = df [ 'Close' ].ewm ( span = 20 , adjust = False ).mean ( )
	df [ 'Momentum' ] = df [ 'Close' ] - df [ 'Close' ].shift ( 10 )
	df [ 'MFI' ] = compute_MFI ( df )
	df [ 'EMA50' ] = df [ 'Close' ].ewm ( span = 50 , adjust = False ).mean ( )
	df [ 'ADX' ] = compute_ADX ( df )
	df [ 'LogRet' ] = np.log ( df [ 'Close' ] / df [ 'Close' ].shift ( 1 ) )
	df [ 'Volatility' ] = df [ 'LogRet' ].rolling ( window = 10 ).std ( )
	df [ 'CCI' ] = compute_CCI ( df )
	df [ 'ATR' ] = compute_ATR ( df )
	df [ 'Stochastic' ] = compute_stochastic ( df )
	df.dropna ( inplace = True )
	
	df [ 'VIX' ] = yf.download ( '^VIX' , start = df.index [ 0 ] , end = df.index [ -1 ] ) [ 'Close' ]
	df [ 'UST10Y' ] = yf.download ( '^TNX' , start = df.index [ 0 ] , end = df.index [ -1 ] ) [ 'Close' ]
	df [ 'USD_Index' ] = yf.download ( 'DX-Y.NYB' , start = df.index [ 0 ] , end = df.index [ -1 ] ) [ 'Close' ]
	return df


# Fetch weather data function
def fetch_weather_data ( api_key , location , start_date , end_date ) :
	url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
	weather_data = [ ]
	for date in pd.date_range ( start_date , end_date ) :
		timestamp = int ( date.timestamp ( ) )
		params = {
			'lat' : location [ 'lat' ] ,
			'lon' : location [ 'lon' ] ,
			'dt' : timestamp ,
			'appid' : api_key
			}
		response = requests.get ( url , params = params )
		if response.status_code == 200 :
			data = response.json ( )
			weather_data.append (
				{
					'date' : date.date ( ) ,
					'temperature' : data [ 'current' ] [ 'temp' ] ,
					'humidity' : data [ 'current' ] [ 'humidity' ] ,
					'pressure' : data [ 'current' ] [ 'pressure' ] ,
					'wind_speed' : data [ 'current' ] [ 'wind_speed' ] ,
					'weather' : data [ 'current' ] [ 'weather' ] [ 0 ] [ 'description' ]
					}
				)
		else :
			logging.warning ( f"Failed to fetch weather data for {date}" )
	
	weather_df = pd.DataFrame ( weather_data )
	
	if 'date' not in weather_df.columns :
		logging.error ( "Weather data missing 'date' column." )
		return pd.DataFrame ( )  # Return empty DataFrame if 'date' column is missing
	
	weather_df [ 'date' ] = pd.to_datetime (
		weather_df [ 'date' ]
		).dt.date  # Ensure the 'date' column is in date format
	
	return weather_df


# Fetch event data function
def fetch_event_data ( api_key , keyword , start_date , end_date ) :
	url = f"https://eventregistry.org/api/v1/event/getEvents"
	params = {
		'query' : keyword ,
		'startDate' : start_date ,
		'endDate' : end_date ,
		'apiKey' : api_key
		}
	response = requests.get ( url , params = params )
	if response.status_code == 200 :
		data = response.json ( )
		events = [ ]
		for event in data [ 'events' ] :
			events.append (
				{
					'date' : pd.to_datetime ( event [ 'eventDate' ] ).date ( ) ,
					'title' : event [ 'title' ] ,
					'description' : event [ 'summary' ] ,
					'location' : event [ 'location' ] [ 'label' ]
					}
				)
		events_df = pd.DataFrame ( events )
		events_df [ 'date' ] = pd.to_datetime (
			events_df [ 'date' ]
			).dt.date  # Ensure the 'date' column is in date format
		if 'date' not in events_df.columns :
			logging.error ( "Event data missing 'date' column." )
			return pd.DataFrame ( )  # Return empty DataFrame if 'date' column is missing
		return events_df
	else :
		logging.warning ( f"Failed to fetch event data from {start_date} to {end_date}" )
		return pd.DataFrame ( )


# Optimize ensemble with Optuna
def optimize_ensemble_with_optuna ( X_train_smote , y_train_smote , X_valid , y_valid , X_test , y_test ) :
	def objective ( trial ) :
		rf_n_estimators = trial.suggest_int ( 'rf_n_estimators' , 50 , 300 )
		rf_max_depth = trial.suggest_int ( 'rf_max_depth' , 3 , 50 )
		rf_min_samples_split = trial.suggest_int ( 'rf_min_samples_split' , 2 , 20 )
		rf_min_samples_leaf = trial.suggest_int ( 'rf_min_samples_leaf' , 1 , 20 )
		xgb_n_estimators = trial.suggest_int ( 'xgb_n_estimators' , 50 , 300 )
		xgb_max_depth = trial.suggest_int ( 'xgb_max_depth' , 3 , 50 )
		xgb_learning_rate = trial.suggest_float ( 'xgb_learning_rate' , 0.01 , 0.3 )
		lgbm_n_estimators = trial.suggest_int ( 'lgbm_n_estimators' , 50 , 300 )
		lgbm_max_depth = trial.suggest_int ( 'lgbm_max_depth' , 3 , 50 )
		lgbm_num_leaves = trial.suggest_int ( 'lgbm_num_leaves' , 31 , 256 )
		lgbm_learning_rate = trial.suggest_float ( 'lgbm_learning_rate' , 0.01 , 0.3 )
		
		rf = RandomForestClassifier (
			n_estimators = rf_n_estimators , max_depth = rf_max_depth , min_samples_split = rf_min_samples_split ,
			min_samples_leaf = rf_min_samples_leaf , random_state = 42
			)
		xgb = XGBClassifier (
			n_estimators = xgb_n_estimators , max_depth = xgb_max_depth , learning_rate = xgb_learning_rate ,
			random_state = 42 , eval_metric = 'mlogloss'
			)
		lgbm = LGBMClassifier (
			n_estimators = lgbm_n_estimators , max_depth = lgbm_max_depth , num_leaves = lgbm_num_leaves ,
			learning_rate = lgbm_learning_rate , random_state = 42
			)
		
		rf.fit ( X_train_smote , y_train_smote )
		xgb.fit (
			X_train_smote , y_train_smote , eval_set = [ (X_valid , y_valid) ] , early_stopping_rounds = 10 ,
			verbose = False
			)
		lgbm.fit (
			X_train_smote , y_train_smote , eval_set = [ (X_valid , y_valid) ] , eval_metric = 'logloss' ,
			callbacks = [ early_stopping ( 10 ) ]
			)
		
		ensemble = VotingClassifier ( estimators = [ ('rf' , rf) , ('xgb' , xgb) , ('lgbm' , lgbm) ] , voting = 'soft' )
		ensemble.fit ( X_train_smote , y_train_smote )
		y_pred = ensemble.predict ( X_valid )
		
		accuracy = accuracy_score ( y_valid , y_pred )
		return accuracy
	
	study = optuna.create_study ( direction = 'maximize' )
	study.optimize ( objective , n_trials = 50 )
	
	best_params = study.best_params
	logging.info ( f"Melhores parâmetros obtidos: {best_params}" )
	
	rf = RandomForestClassifier (
		n_estimators = best_params [ 'rf_n_estimators' ] , max_depth = best_params [ 'rf_max_depth' ] ,
		min_samples_split = best_params [ 'rf_min_samples_split' ] ,
		min_samples_leaf = best_params [ 'rf_min_samples_leaf' ] ,
		random_state = 42
		)
	xgb = XGBClassifier (
		n_estimators = best_params [ 'xgb_n_estimators' ] , max_depth = best_params [ 'xgb_max_depth' ] ,
		learning_rate = best_params [ 'xgb_learning_rate' ] , random_state = 42 , eval_metric = 'mlogloss'
		)
	lgbm = LGBMClassifier (
		n_estimators = best_params [ 'lgbm_n_estimators' ] , max_depth = best_params [ 'lgbm_max_depth' ] ,
		num_leaves = best_params [ 'lgbm_num_leaves' ] , learning_rate = best_params [ 'lgbm_learning_rate' ] ,
		random_state = 42
		)
	
	rf.fit ( X_train_smote , y_train_smote )
	xgb.fit (
		X_train_smote , y_train_smote , eval_set = [ (X_valid , y_valid) ] , early_stopping_rounds = 10 ,
		verbose = False
		)
	lgbm.fit (
		X_train_smote , y_train_smote , eval_set = [ (X_valid , y_valid) ] , eval_metric = 'logloss' ,
		callbacks = [ early_stopping ( 10 ) ]
		)
	
	best_ensemble = VotingClassifier (
		estimators = [ ('rf' , rf) , ('xgb' , xgb) , ('lgbm' , lgbm) ] , voting = 'soft'
		)
	best_ensemble.fit ( X_train_smote , y_train_smote )
	y_pred = best_ensemble.predict ( X_test )
	
	accuracy = accuracy_score ( y_test , y_pred )
	precision = precision_score ( y_test , y_pred , average = 'weighted' , zero_division = 0 )
	recall = recall_score ( y_test , y_pred , average = 'weighted' , zero_division = 0 )
	f1 = f1_score ( y_test , y_pred , average = 'weighted' , zero_division = 0 )
	
	try :
		roc_auc = roc_auc_score (
			label_binarize ( y_test , classes = [ 0 , 1 , 2 ] ) ,
			best_ensemble.predict_proba ( X_test ) , multi_class = 'ovr'
			)
	except ValueError as e :
		logging.warning ( f"ROC AUC score could not be calculated: {e}" )
		roc_auc = None
	
	return best_ensemble , accuracy , precision , recall , f1 , roc_auc


# Download data and calculate indicators
@st.cache_data
def baixar_dados ( ticker , start_date , end_date ) :
	try :
		logging.info ( f"Baixando dados para {ticker} de {start_date} até {end_date}..." )
		df = yf.download ( ticker , start = start_date , end = end_date )
		if df.empty :
			logging.error ( f"No data available for {ticker}." )
			return None
		if df.isnull ( ).values.any ( ) :
			logging.warning ( "Dados incompletos foram encontrados e removidos." )
			df.dropna ( inplace = True )
		df = calcular_indicadores ( df )
		logging.info ( f"Dados para {ticker} baixados e indicadores calculados com sucesso." )
		return df
	except Exception as e :
		error_message = f"Erro ao baixar dados para {ticker}: {e}"
		logging.error ( error_message )
		send_email_alert ( "Data Download Error" , str ( e ) )
		st.error ( error_message )
		return None


# Sentiment analysis function
def analyze_sentiment ( ticker , date , api_tool , api_key = None , use_social_searcher = False ) :
	global response
	if api_tool == "newsapi" and api_key :
		url = f"https://newsapi.org/v2/everything?q={ticker}&from={date}&sortBy=popularity&apiKey={api_key}"
		try :
			response = requests.get ( url )
			response.raise_for_status ( )
			data = response.json ( )
			if 'articles' not in data :
				error_message = f"The response does not contain 'articles'. Full response: {data}"
				logging.error ( error_message )
				st.error ( error_message )
				return 0
			
			sentiments = [ np.random.choice ( [ 'positive' , 'negative' , 'neutral' ] ) for _ in data [ 'articles' ] ]
			return sum ( 1 for s in sentiments if s == 'positive' ) / len ( sentiments )
		except requests.exceptions.HTTPError as http_err :
			if response.status_code == 426 :
				error_message = "News API key requires an upgrade."
				logging.error ( error_message )
				st.error ( error_message )
			else :
				error_message = f"HTTP error occurred: {http_err}"
				logging.error ( error_message )
				st.error ( error_message )
			return 0
		except requests.exceptions.RequestException as e :
			error_message = f"Error calling News API: {e}"
			logging.error ( error_message )
			st.error ( error_message )
			return 0
	elif use_social_searcher :
		url = f"https://www.social-searcher.com/{ticker}"
		try :
			response = requests.get ( url )
			response.raise_for_status ( )
			soup = BeautifulSoup ( response.text , 'html.parser' )
			sentiment_data = soup.find ( 'div' , { 'class' : 'item sentiment' } )
			if sentiment_data :
				positive = sentiment_data.find ( 'span' , { 'class' : 'positive' } )
				negative = sentiment_data.find ( 'span' , { 'class' : 'negative' } )
				if positive and negative :
					positive_value = int ( positive.text.strip ( '%' ) )
					negative_value = int ( negative.text.strip ( '%' ) )
					sentiment_score = positive_value / (positive_value + negative_value)
					return sentiment_score
			return 0
		except requests.exceptions.RequestException as e :
			error_message = f"Erro ao chamar a Social Mention: {e}"
			logging.error ( error_message )
			st.error ( error_message )
			return 0
	else :
		logging.info ( "Nenhuma opção de análise de sentimento foi fornecida." )
		return 0


# Dynamic data labeling function
def etiquetar_dados_dinamicamente ( df , vix_value , sentimento ) :
	df [ 'Etiqueta' ] = 'Manter'
	if vix_value > 25 :
		df.loc [ (df [ 'RSI' ] > 70) & (df [ 'MFI' ] > 80) , 'Etiqueta' ] = 'Venda'
		df.loc [ (df [ 'RSI' ] < 30) & (df [ 'MFI' ] < 20) , 'Etiqueta' ] = 'Compra'
	else :
		df.loc [ (df [ 'RSI' ] > 65) & (df [ 'MFI' ] > 75) , 'Etiqueta' ] = 'Venda'
		df.loc [ (df [ 'RSI' ] < 35) & (df [ 'MFI' ] < 25) , 'Etiqueta' ] = 'Compra'
	
	if sentimento > 0.6 :
		df.loc [ (df [ 'Etiqueta' ] == 'Manter') & (df [ 'RSI' ] < 50) , 'Etiqueta' ] = 'Compra'
	elif sentimento < 0.4 :
		df.loc [ (df [ 'Etiqueta' ] == 'Manter') & (df [ 'RSI' ] > 50) , 'Etiqueta' ] = 'Venda'
	
	if 'Compra' not in df [ 'Etiqueta' ].values :
		df = pd.concat ( [ df , pd.DataFrame ( { 'Etiqueta' : [ 'Compra' ] } ) ] , ignore_index = True )
	if 'Venda' not in df [ 'Etiqueta' ].values :
		df = pd.concat ( [ df , pd.DataFrame ( { 'Etiqueta' : [ 'Venda' ] } ) ] , ignore_index = True )
	if 'Manter' not in df [ 'Etiqueta' ].values :
		df = pd.concat ( [ df , pd.DataFrame ( { 'Etiqueta' : [ 'Manter' ] } ) ] , ignore_index = True )
	
	return df


# Suggest allocation function
def sugerir_alocacao ( ticker , probabilidades , tolerancia_risco , volatilidade ) :
	prob_compra = probabilidades [ 1 ]
	prob_venda = probabilidades [ 2 ]
	prob_manter = probabilidades [ 0 ]
	sugestoes = {
		'comprar' : "Não recomendado conforme sua escolha de tolerância ao risco." ,
		'vender' : "Não recomendado conforme sua escolha de tolerância ao risco." ,
		'manter' : "Recomendado conforme sua escolha de tolerância ao risco."
		}
	
	if tolerancia_risco == 'Alta' :
		if prob_compra > 0.7 :
			sugestoes [ 'comprar' ] = f"Alocar 20% do portfólio para {ticker}, Stop Loss: 85%, Take Profit: 150%"
		elif prob_compra > 0.5 :
			sugestoes [ 'comprar' ] = f"Alocar 10% do portfólio para {ticker}, Stop Loss: 90%, Take Profit: 140%"
		if prob_venda > 0.7 :
			sugestoes [ 'vender' ] = f"Reduzir 20% do portfólio em {ticker}, Stop Loss: 105%, Take Profit: 70%"
		elif prob_venda > 0.5 :
			sugestoes [ 'vender' ] = f"Reduzir 10% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 75%"
	
	elif tolerancia_risco == 'Média' :
		if prob_compra > 0.6 :
			sugestoes [ 'comprar' ] = f"Alocar 15% do portfólio para {ticker}, Stop Loss: 90%, Take Profit: 130%"
		elif prob_compra > 0.4 :
			sugestoes [ 'comprar' ] = f"Alocar 7% do portfólio para {ticker}, Stop Loss: 95%, Take Profit: 120%"
		if prob_venda > 0.6 :
			sugestoes [ 'vender' ] = f"Reduzir 15% do portfólio em {ticker}, Stop Loss: 102%, Take Profit: 75%"
		elif prob_venda > 0.4 :
			sugestoes [ 'vender' ] = f"Reduzir 7% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 80%"
	
	elif tolerancia_risco == 'Baixa' :
		if prob_compra > 0.5 :
			sugestoes [ 'comprar' ] = f"Alocar 10% do portfólio para {ticker}, Stop Loss: 95%, Take Profit: 110%"
		elif prob_compra > 0.3 :
			sugestoes [ 'comprar' ] = f"Alocar 5% do portfólio para {ticker}, Stop Loss: 98%, Take Profit: 105%"
		if prob_venda > 0.5 :
			sugestoes [ 'vender' ] = f"Reduzir 10% do portfólio em {ticker}, Stop Loss: 101%, Take Profit: 80%"
		elif prob_venda > 0.3 :
			sugestoes [ 'vender' ] = f"Reduzir 5% do portfólio em {ticker}, Stop Loss: 100%, Take Profit: 85%"
	
	if volatilidade > 0.02 :
		sugestoes [
			'manter' ] = f"Volatilidade alta detectada, sugere-se manter {ticker} e evitar novas posições até que a volatilidade diminua."
	
	return sugestoes


@contextmanager
def temporary_files_cleanup ( ) :
	temp_files = [ ]
	yield temp_files
	for file in temp_files :
		try :
			os.remove ( file )
		except OSError as e :
			logging.warning ( f"Failed to remove temporary file {file}: {e}" )


# Process a stock
def processar_acao ( ticker , start_date , end_date , tolerancia_risco , api_tool , api_key , use_social_searcher ,
                     weather_api_key , location , event_api_key ) :
	dados_acao = baixar_dados ( ticker , start_date , end_date )
	if dados_acao is None :
		st.error ( f"No data available for {ticker}." )
		return None , None , None , False
	
	if 'VIX' not in dados_acao.columns or dados_acao [ 'VIX' ].isnull ( ).all ( ) :
		logging.error ( f"Missing VIX data for {ticker}." )
		st.error ( f"Missing VIX data for {ticker}." )
		return None , None , None , False
	
	vix_value = dados_acao [ 'VIX' ].iloc [ -1 ]
	if len ( dados_acao ) < 30 :
		logging.error ( f"Insufficient data for sentiment analysis for {ticker}." )
		st.error ( f"Insufficient data for sentiment analysis for {ticker}." )
		return None , None , None , False
	
	sentimento = analyze_sentiment ( ticker , dados_acao.index [ -30 ] , api_tool , api_key , use_social_searcher )
	use_sentiment = api_key or use_social_searcher
	
	dados_acao = etiquetar_dados_dinamicamente ( dados_acao , vix_value , sentimento )
	
	logging.info ( f"Distribuição das classes para {ticker}:\n{dados_acao [ 'Etiqueta' ].value_counts ( )}" )
	
	features = [
		'MA20' , 'MA50' , 'RSI' , 'MACD' , 'MACD_Signal' , 'BB_Upper' , 'BB_Lower' , 'MA10' , 'MA100' ,
		'EMA20' , 'Momentum' , 'MFI' , 'EMA50' , 'ADX' , 'LogRet' , 'Volatility' , 'CCI' , 'ATR' ,
		'Stochastic' , 'VIX' , 'UST10Y' , 'USD_Index'
		]
	
	if weather_api_key and location :
		weather_data = fetch_weather_data ( weather_api_key , location , start_date , end_date )
		if not weather_data.empty :
			dados_acao = dados_acao.merge (
				weather_data , left_on = dados_acao.index.date , right_on = 'date' , how = 'left'
				)
			features.extend ( [ 'temperature' , 'humidity' , 'pressure' , 'wind_speed' ] )
	
	if event_api_key :
		event_data = fetch_event_data ( event_api_key , ticker , start_date , end_date )
		if not event_data.empty :
			dados_acao = dados_acao.merge (
				event_data , left_on = dados_acao.index.date , right_on = 'date' , how = 'left'
				)
			features.extend ( [ 'title' , 'description' , 'location' ] )  # Adjust according to the actual data needed
	
	X = dados_acao [ features ]
	y = dados_acao [ 'Etiqueta' ]
	
	label_mapping = { 'Compra' : 0 , 'Manter' : 1 , 'Venda' : 2 }
	y = y.map ( label_mapping )
	
	scaler = StandardScaler ( )
	X = scaler.fit_transform ( X )
	
	present_classes = set ( y )
	expected_classes = set ( label_mapping.values ( ) )
	
	if not present_classes.issuperset ( expected_classes ) :
		missing_classes = expected_classes - present_classes
		error_message = f"Classes {missing_classes} are not present in the data for {ticker}. Model performance may be affected."
		logging.warning ( error_message )
		st.warning ( error_message )
	
	tscv = TimeSeriesSplit ( n_splits = 5 )
	for train_index , test_index in tscv.split ( X , y ) :
		X_train , X_test = X [ train_index ] , X [ test_index ]
		y_train , y_test = y.iloc [ train_index ] , y.iloc [ test_index ]
		if len ( set ( y_train ) ) == len ( present_classes ) and len ( set ( y_test ) ) == len ( present_classes ) :
			break
	
	if np.isnan ( X_train ).any ( ) or np.isnan ( X_test ).any ( ) :
		error_message = "Dados de treinamento ou teste contêm NaNs. Lidando com valores ausentes."
		logging.error ( error_message )
		st.error ( error_message )
		X_train = np.nan_to_num ( X_train )
		X_test = np.nan_to_num ( X_test )
	
	class_counts = y_train.value_counts ( )
	minority_class = class_counts.idxmin ( )
	n_minority_samples = class_counts.min ( )
	
	if n_minority_samples < 2 :
		error_message = f"Not enough samples for minority class {minority_class} to apply SMOTE."
		logging.warning ( error_message )
		st.warning ( error_message )
		X_train_smote , y_train_smote = X_train , y_train
	else :
		smote_k = min ( 5 , n_minority_samples - 1 )
		smote = SMOTE ( random_state = 42 , k_neighbors = smote_k )
		X_train_smote , y_train_smote = smote.fit_resample ( X_train , y_train )
	
	if n_minority_samples < 2 :
		X_train_smote , X_valid , y_train_smote , y_valid = train_test_split (
			X_train_smote , y_train_smote , test_size = 0.2 , random_state = 42
			)
	else :
		X_train_smote , X_valid , y_train_smote , y_valid = train_test_split (
			X_train_smote , y_train_smote , test_size = 0.2 , stratify = y_train_smote , random_state = 42
			)
	
	best_model , accuracy , precision , recall , f1 , roc_auc = optimize_ensemble_with_optuna (
		X_train_smote , y_train_smote , X_valid , y_valid , X_test , y_test
		)
	
	logging.info ( f"Melhor precisão obtida: {accuracy}" )
	
	explainer = shap.TreeExplainer ( best_model.named_estimators_ [ 'xgb' ] )
	shap_values = explainer.shap_values ( X_test )
	
	plt.figure ( )
	if isinstance ( shap_values , list ) :
		shap.summary_plot ( shap_values [ 1 ] , X_test , plot_type = "bar" , show = False )
	else :
		shap.summary_plot ( shap_values , X_test , plot_type = "bar" , show = False )
	shap_plot_path = "shap_summary_plot.png"
	plt.savefig ( shap_plot_path )
	logging.info ( f"SHAP summary plot saved at {shap_plot_path}" )
	plt.close ( )
	
	dados_acao = dados_acao.asfreq ( 'D' )
	
	arima_model = ARIMA ( dados_acao [ 'Close' ] , order = (1 , 1 , 1) )
	arima_results = arima_model.fit ( )
	arima_forecast = arima_results.forecast ( steps = 30 )
	
	if len ( arima_forecast ) > 0 :
		logging.info ( f"ARIMA Previsão (30 dias): {arima_forecast.iloc [ -1 ]}" )
	else :
		logging.warning ( "ARIMA não conseguiu prever." )
	
	with temporary_files_cleanup ( ) as temp_files :
		prophet_model = Prophet ( daily_seasonality = True )
		prophet_df = pd.DataFrame ( { 'ds' : dados_acao.index , 'y' : dados_acao [ 'Close' ] } )
		prophet_model.fit ( prophet_df )
		future = prophet_model.make_future_dataframe ( periods = 30 )
		prophet_forecast = prophet_model.predict ( future )
		
		temp_file_path = tempfile.mkstemp ( ) [ 1 ]
		temp_files.append ( temp_file_path )
		with open ( temp_file_path , 'w' ) as f :
			f.write ( prophet_forecast.to_json ( ) )
	
	if not prophet_forecast.empty :
		logging.info ( f"Prophet Previsão (30 dias): {prophet_forecast [ 'yhat' ].iloc [ -1 ]}" )
	else :
		logging.warning ( "Prophet não conseguiu prever." )
	
	X_recent = X [ -30 : ]
	probabilities = best_model.predict_proba ( X_recent )
	avg_probabilities = np.mean ( probabilities , axis = 0 )
	sugestoes = sugerir_alocacao (
		ticker , avg_probabilities , tolerancia_risco , dados_acao [ 'Volatility' ].iloc [ -30 : ].mean ( )
		)
	
	return best_model , accuracy , sugestoes , use_sentiment


# Fetch conversion rate function
@st.cache_data
def fetch_conversion_rate ( from_currency , to_currency ) :
	url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
	try :
		response = requests.get ( url )
		response.raise_for_status ( )
		data = response.json ( )
		rates = data [ 'rates' ]
		if to_currency in rates :
			return rates [ to_currency ]
		else :
			error_message = f"Currency {to_currency} not found in the exchange rates."
			logging.error ( error_message )
			st.error ( error_message )
			return None
	except Exception as e :
		error_message = f"Failed to fetch exchange rate: {e}"
		logging.error ( error_message )
		send_email_alert ( "Exchange Rate Fetch Error" , str ( e ) )
		st.error ( error_message )
		return None


# Currency conversion function
def convert_currency ( amount , rate ) :
	return amount * rate


# Translation dictionary
translations = {
	'en' : {
		'title' : "Stock Prediction" ,
		'start_date' : "Start Date" ,
		'end_date' : "End Date" ,
		'risk_tolerance' : "Risk Tolerance" ,
		'select_tickers' : "Select Tickers (Up to 5)" ,
		'api_tool' : "Choose Sentiment Analysis Tool" ,
		'api_key' : "API Key for {} (Optional)" ,
		'use_social_searcher' : "Use Social Mention for Sentiment Analysis" ,
		'weather_api_key' : "API Key for OpenWeatherMap" ,
		'location' : "Location (Latitude, Longitude)" ,
		'event_api_key' : "API Key for EventRegistry" ,
		'currency' : "Select Currency" ,
		'total_portfolio' : "Total Portfolio" ,
		'process' : "Process" ,
		'warning_tickers' : "You can add up to 5 tickers." ,
		'alert_title' : "Data Download Error" ,
		'alert_body' : "Error downloading data for {}: {}" ,
		'calculation_title' : "Calculation for {}" ,
		'allocation_title' : "Allocation for {}" ,
		'buy_allocation' : "Buy Allocation" ,
		'reduce_allocation' : "Reduce Allocation" ,
		'keep_allocation' : "Keep Allocation" ,
		'current_price' : "Current Price of {}: {} {:.2f}" ,
		'allocation_formula' : "**Formula:** Allocation = (Buy Percentage / 100) * Total Portfolio" ,
		'calculation_formula' : "**Calculation:** ({:.2f} / 100) * {:.2f} = {:.2f} USD = {:.2f} {}" ,
		'stock_quantity' : "Quantity of Stocks: {:.2f}" ,
		'stop_loss' : "Stop Loss: {} {:.2f}" ,
		'take_profit' : "Take Profit: {} {:.2f}" ,
		'conversion_failed' : "Failed to fetch conversion rate for {}." ,
		'warning_message' : "**WARNING**: Past performance does not guarantee future results. The stock market involves substantial risks. Consult a financial advisor before making investment decisions."
		} ,
	'pt' : {
		'title' : "Previsão de Ações" ,
		'start_date' : "Data de Início" ,
		'end_date' : "Data de Fim" ,
		'risk_tolerance' : "Tolerância ao Risco" ,
		'select_tickers' : "Adicionar Tickers (Até 5)" ,
		'api_tool' : "Escolha a ferramenta de análise de sentimento" ,
		'api_key' : "Chave API para {} (Opcional)" ,
		'use_social_searcher' : "Usar Social Mention para Análise de Sentimento" ,
		'weather_api_key' : "Chave API para OpenWeatherMap" ,
		'location' : "Localização (Latitude, Longitude)" ,
		'event_api_key' : "Chave API para EventRegistry" ,
		'currency' : "Selecionar moeda" ,
		'total_portfolio' : "Total do Portfólio" ,
		'process' : "Processar" ,
		'warning_tickers' : "Você pode adicionar até 5 tickers." ,
		'alert_title' : "Erro ao Baixar Dados" ,
		'alert_body' : "Erro ao baixar dados para {}: {}" ,
		'calculation_title' : "Cálculo para {}" ,
		'allocation_title' : "Alocação para {}" ,
		'buy_allocation' : "Alocação de Compra" ,
		'reduce_allocation' : "Reduzir Alocação" ,
		'keep_allocation' : "Manter Alocação" ,
		'current_price' : "Preço Atual de {}: {} {:.2f}" ,
		'allocation_formula' : "**Fórmula:** Alocar = (Percentual de Compra / 100) * Total do Portfólio" ,
		'calculation_formula' : "**Cálculo:** ({:.2f} / 100) * {:.2f} = {:.2f} USD = {:.2f} {}" ,
		'stock_quantity' : "Quantidade de Ações: {:.2f}" ,
		'stop_loss' : "Stop Loss: {} {:.2f}" ,
		'take_profit' : "Take Profit: {} {:.2f}" ,
		'conversion_failed' : "Falha ao buscar taxa de conversão para {}." ,
		'warning_message' : "**AVISO**: O desempenho passado não garante resultados futuros. O mercado de ações envolve riscos substanciais. Consulte um conselheiro financeiro antes de tomar decisões de investimento."
		}
	# Add more languages as needed
	}

# Add language selection to the sidebar
language = st.sidebar.selectbox ( "Select Language" , options = [ "en" , "pt" ] )

# Get the translations for the selected language
t = translations [ language ]


# Main function
def main ( ) :
	st.title ( t [ 'title' ] )
	st.markdown ( t [ 'warning_message' ] )
	
	st.sidebar.header ( "Configuração" )
	
	config_path = 'config.json'
	config = load_config ( config_path )
	start_date = st.sidebar.date_input (
		t [ 'start_date' ] , pd.to_datetime ( '2000-01-01' ).date ( ) ,
		min_value = pd.to_datetime ( '2000-01-01' ).date ( ) , max_value = pd.Timestamp.now ( ).date ( )
		)
	end_date = st.sidebar.date_input (
		t [ 'end_date' ] , pd.Timestamp.now ( ).date ( ) , min_value = pd.to_datetime ( '2000-01-01' ).date ( ) ,
		max_value = pd.Timestamp.now ( ).date ( )
		)
	tolerancia_risco = st.sidebar.selectbox ( t [ 'risk_tolerance' ] , [ 'Alta' , 'Média' , 'Baixa' ] )
	
	with open ( 'tikers.txt' , 'r' ) as f :
		all_tickers = [ line.strip ( ) for line in f ]
	
	adicionar_tickers = st.sidebar.multiselect ( t [ 'select_tickers' ] , all_tickers , default = all_tickers [ :5 ] )
	
	api_tool = st.sidebar.selectbox (
		t [ 'api_tool' ] , [ 'newsapi' , 'ravenpack' , 'bloomberg' , 'sentdex' , 'social_searcher' ]
		)
	api_key_input = st.sidebar.text_input ( t [ 'api_key' ].format ( api_tool ) )
	use_social_searcher = st.sidebar.checkbox ( t [ 'use_social_searcher' ] )
	
	weather_api_key = st.sidebar.text_input ( t [ 'weather_api_key' ] )
	location = st.sidebar.text_input ( t [ 'location' ] , "(latitude, longitude)" )
	
	# Check if location is valid
	if location and location != "(latitude, longitude)" :
		try :
			lat , lon = map ( float , location.strip ( '()' ).split ( ',' ) )
			location = { 'lat' : lat , 'lon' : lon }
		except ValueError :
			st.error ( "Invalid location format. Please enter as (latitude, longitude)." )
			location = None
	else :
		location = None
	
	event_api_key = st.sidebar.text_input ( t [ 'event_api_key' ] )
	
	selected_currency = st.sidebar.selectbox ( t [ 'currency' ] , [ "USD" , "BRL" , "EUR" , "GBP" ] )
	total_portfolio = st.sidebar.number_input ( t [ 'total_portfolio' ] , min_value = 0.0 , value = 100000.0 )
	
	if st.sidebar.button ( t [ 'process' ] ) :
		if len ( adicionar_tickers ) > 5 :
			st.sidebar.warning ( t [ 'warning_tickers' ] )
		else :
			valid_tickers = validar_tickers ( adicionar_tickers )
			modelos = { }
			
			for ticker in valid_tickers :
				modelo , acuracia , sugestoes , use_sentiment = processar_acao (
					ticker , start_date.strftime ( '%Y-%m-%d' ) , end_date.strftime ( '%Y-%m-%d' ) , tolerancia_risco ,
					api_tool , api_key_input , use_social_searcher , weather_api_key , location , event_api_key
					)
				if modelo and sugestoes :
					modelos [ ticker ] = modelo
					acuracia_percent = round ( acuracia * 100 , 2 )
					if use_sentiment :
						title = f"{ticker} ({t [ 'calculation_title' ].split ( ) [ 0 ]} {acuracia_percent}%)"
					else :
						title = f"{ticker} ({t [ 'calculation_title' ].split ( ) [ 0 ]} {acuracia_percent}%)"
					st.write ( f"### {title}" )
					st.write ( f"**{t [ 'buy_allocation' ]}:**\n- {sugestoes [ 'comprar' ]}" )
					st.write ( f"**{t [ 'reduce_allocation' ]}:**\n- {sugestoes [ 'vender' ]}" )
					st.write ( f"**{t [ 'keep_allocation' ]}:**\n- {sugestoes [ 'manter' ]}" )
					
					current_price = yf.Ticker ( ticker ).history ( period = '1d' ) [ 'Close' ].iloc [ 0 ]
					st.write ( t [ 'current_price' ].format ( ticker , selected_currency , current_price ) )
					
					conversion_rate = fetch_conversion_rate ( 'USD' , selected_currency )
					if conversion_rate is None :
						st.error ( t [ 'conversion_failed' ].format ( selected_currency ) )
						continue
					
					st.write ( f"## {t [ 'calculation_title' ].format ( ticker )}" )
					st.write ( f"### {t [ 'buy_allocation' ]}" )
					if "não recomendado" not in sugestoes [ 'comprar' ].lower ( ) :
						compra_percent = float ( sugestoes [ 'comprar' ].split ( ' ' ) [ 1 ].strip ( '%' ) )
						alocar_compra_usd = (compra_percent / 100) * total_portfolio
						alocar_compra = convert_currency ( alocar_compra_usd , conversion_rate )
						st.write (
							f"{t [ 'allocation_title' ].split ( ) [ 0 ]}: {selected_currency} {alocar_compra:.2f}"
							)
						st.write ( t [ 'allocation_formula' ] )
						st.write (
							t [ 'calculation_formula' ].format (
								compra_percent , total_portfolio , alocar_compra_usd , alocar_compra , selected_currency
								)
							)
						
						quantidade_acoes = alocar_compra / current_price
						st.write ( t [ 'stock_quantity' ].format ( quantidade_acoes ) )
						stop_loss_compra = current_price * 0.98
						take_profit_compra = current_price * 1.05
						st.write ( t [ 'stop_loss' ].format ( selected_currency , stop_loss_compra ) )
						st.write ( t [ 'take_profit' ].format ( selected_currency , take_profit_compra ) )
					
					st.write ( f"### {t [ 'reduce_allocation' ]}" )
					if "não recomendado" not in sugestoes [ 'vender' ].lower ( ) :
						venda_percent = float ( sugestoes [ 'vender' ].split ( ' ' ) [ 1 ].strip ( '%' ) )
						alocar_venda_usd = (venda_percent / 100) * total_portfolio
						alocar_venda = convert_currency ( alocar_venda_usd , conversion_rate )
						st.write (
							f"{t [ 'allocation_title' ].split ( ) [ 1 ]}: {selected_currency} {alocar_venda:.2f}"
							)
						st.write ( t [ 'allocation_formula' ] )
						st.write (
							t [ 'calculation_formula' ].format (
								venda_percent , total_portfolio , alocar_venda_usd , alocar_venda , selected_currency
								)
							)
						
						quantidade_venda = alocar_venda / current_price
						st.write ( t [ 'stock_quantity' ].format ( quantidade_venda ) )
						stop_loss_venda = current_price * 1.01
						take_profit_venda = current_price * 0.80
						st.write ( t [ 'stop_loss' ].format ( selected_currency , stop_loss_venda ) )
						st.write ( t [ 'take_profit' ].format ( selected_currency , take_profit_venda ) )
			
			with open ( "conselhos.txt" , "w" ) as file :
				for ticker in valid_tickers :
					if ticker in modelos :
						conselho = f"{ticker}: Comprar: {sugestoes [ 'comprar' ]} Vender: {sugestoes [ 'vender' ]} Manter: {sugestoes [ 'manter' ]}"
						file.write ( conselho + "\n" )
			
			os.system ( "notepad.exe conselhos.txt" )


if __name__ == "__main__" :
	main ( )
