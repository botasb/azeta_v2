from threading import local
import shap
import streamlit as st
import streamlit.components.v1 as components
import catboost
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import catboost
from catboost import CatBoostRegressor, Pool
from datetime import datetime
import random
import plotly.graph_objs as go
from io import StringIO
from cryptography.fernet import Fernet



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


@st.cache
def load_derived():
    #return pd.read_csv('./data/azeta_train_derived_v4.tar.xz')
    return pd.read_csv('./data/azeta_train_derived_v5_before_train.tar.xz').rename(columns={'azeta_train_derived_v5_before_train.csv':'Libro'})




@st.cache
def process_data(local_df, cols_input, cols_empty, manual=False):
    local_df.rename(columns=mes_columns_dict, inplace=True)

    if manual:
        #st.text(type(local_df))
        local_df['Mes 22'] = mes_22
        local_df['Mes 23'] = mes_23
        local_df['Mes 24'] = mes_24
        local_df['Libro'] = libro
        local_df['Autor'] = autor
        local_df['Materia'] = materia
        local_df['Editorial'] = editorial
        local_df['Coleccion'] = coleccion
        local_df['Ptes_Proveedor'] = ptes_proveedor
        local_df['Ptes_Clientes'] = ptes_clientes
        local_df['Stock'] = stock
        local_df['Pedido real'] = pedido_real

    if 'Date' in local_df.columns:
        local_df['Date'] = pd.to_datetime(local_df['Date'], errors ='coerce')
        #st.text(local_df['Date'])
    #st.text(type(local_df['Date']))

    #else:
        local_df['Dia_pedido'] = local_df['Date'].dt.day
        local_df['Mes_pedido'] = local_df['Date'].dt.month
        local_df['Semana_pedido'] = local_df['Date'].dt.isocalendar().week
        #local_df.drop(columns=['Date'], inplace=True)
        local_df['Date'] = local_df['Date'].astype('str')


    for c in ['Libro',
        'Tipo Articulo',
        'Editorial',
        'Coleccion',
        'Autor',
        'Proveedor',
        'Materia',
        'Idioma',
        'Date'
        ]:
        if c in cols_input:
            local_df[c] = local_df[c].astype('str')
            local_df[c] = local_df[c].str.strip()
            local_df[c] = local_df[c].replace("", "DESCONOCIDO")



    if materia=='':
        #st.text(local_df.Materia)
        local_df[['Materia']].fillna('DESCONOCIDO', inplace=True)
        #st.text(local_df.Materia)
    for ix,c in local_df.dtypes.iteritems():
        if c=='int64':
            local_df[ix] = local_df[ix].astype(np.int16)

    if 'Pedido_real' not in local_df.columns:
        local_df['Pedido_real'] = local_df['Pedido real']
        local_df['Pedido real'] = np.log1p(local_df['Pedido real'])

    for ix,c in local_df.dtypes.iteritems():
        if (ix not in ['Pedido_real','Pedido real']) and (c=='int16'):
            local_df['log_' + str(ix)] = np.log1p(local_df[ix])

    for c in ['Dia_pedido','Mes_pedido','Semana_pedido']:
        local_df[c] = local_df[[c]].astype(np.int16)

    for i in range(3,26):
        local_df['Acum_Mes ' + str(i-1)] = local_df[['Mes ' + str(j) for j in range(1,i)]].sum(axis=1)

    for c in mes_columns + acum_mes_columns + ['Stock','Ptes_Clientes','Ptes_Proveedor']:
        local_df['log_'+str(c)] = np.log1p(local_df[c])

    # Load trained 'derived averages', TO DO - Load the pivot table directly
    #previous_trained_local_df = pd.read_csv('./data/azeta_train_derived_v4.tar.xz')
    previous_trained_local_df = load_derived()

    table = pd.pivot_table(previous_trained_local_df, values='Pedido_real', index=['Editorial', 'Coleccion', 'Proveedor', 'Materia'],
                    columns=[], aggfunc=np.mean)
    for c in ['Libro', 'Autor','Editorial','Coleccion', 'Proveedor']:
        for col_train in local_df[[c]][c]:
            # Si esa selección está en los pond_avg de train, usar ese dato
            if  len(previous_trained_local_df.loc[previous_trained_local_df[c]==col_train,'pond_avg_'+str(c)].unique())>0:
                local_df.loc[local_df[c]==col_train,'pond_avg_'+str(c)] = previous_trained_local_df.loc[previous_trained_local_df[c]==col_train,'pond_avg_'+str(c)].unique()[0]
            else:
                # Sino, coger como trainor la media de un libro de ese ['Editorial', 'Coleccion', 'Proveedor', 'Materia'] si lo hay, o la media
                idx = pd.IndexSlice
                try:
                    tmp_par = dict(local_df[local_df[c]==col_train][['Editorial','Coleccion','Proveedor','Materia']].iloc[0])
                    tmp_train = table.loc[idx[tmp_par['Editorial'],tmp_par['Coleccion'],tmp_par['Proveedor'],tmp_par['Materia']],'Pedido_real'].mean()
                except:
                    try:
                        tmp_train = table.loc[idx[tmp_par['Editorial'],tmp_par['Coleccion'],tmp_par['Proveedor'],:],'Pedido_real'].mean()
                    except:
                        try:
                            tmp_train = table.loc[idx[:,tmp_par['Coleccion'],tmp_par['Proveedor'],:],'Pedido_real'].mean()
                        except:
                            try:
                                tmp_train = table.loc[idx[tmp_par['Editorial'],:,tmp_par['Proveedor'],:],'Pedido_real'].mean()
                            except:
                                try:
                                    tmp_train = table.loc[idx[:,:,tmp_par['Proveedor'],:],'Pedido_real'].mean()
                                except:
                                    tmp_train = previous_trained_local_df['pond_avg_'+str(c)].mean()
                local_df.loc[local_df[c]==col_train,'pond_avg_'+str(c)] = tmp_train

    for c in ['Tipo Articulo', 'Fecha']:
        if c in local_df.columns:
            local_df.drop(columns=[c], inplace=True)
    for c in ['Mes 1','Mes 2','Mes 3','Mes 4','Mes 5','Mes 6','Mes 7','Mes 8','Mes 9','Mes 10','Mes 11','Mes 12',
            'Mes 13','Mes 14','Mes 15','Mes 16','Mes 17','Mes 18','Mes 19','Mes 20','Mes 21' ,'Mes 22','Mes 23',
            'Mes 24',
            'Acum_Mes 2','Acum_Mes 3','Acum_Mes 4','Acum_Mes 5','Acum_Mes 6','Acum_Mes 7','Acum_Mes 8',
            'Acum_Mes 9','Acum_Mes 10','Acum_Mes 11','Acum_Mes 12', 'Acum_Mes 13','Acum_Mes 14',
            'Acum_Mes 15','Acum_Mes 16','Acum_Mes 17','Acum_Mes 18','Acum_Mes 19','Acum_Mes 20','Acum_Mes 21'
            ,'Acum_Mes 22','Acum_Mes 23','Acum_Mes 24']:
        local_df[c] = local_df[c].astype('int32')

    for ix,c in local_df.dtypes.iteritems():
        if c in ['float64']:
            local_df[ix] = local_df[ix].astype('float32')
        if c in ['int64']:
            local_df[ix] = local_df[ix].astype('int32')
    return local_df


@st.cache
def get_model():
    # Model
    model = catboost.CatBoostRegressor()
    model.load_model('./data/azeta_gpu_dump_real_gs')
    return model


@st.cache(allow_output_mutation=True)
def get_sample(df, num):
    return df.sample(num).reset_index(drop=True)

@st.cache
def load_data(key):
    #os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


    # using the key

    fernet = Fernet(key)

    # opening the encrypted file
    with open('./data/ml_azeta_prepro_sample_v3_crypted.csv', 'rb') as enc_file:
        encrypted = enc_file.read()

    # decrypting the file
    decrypted = fernet.decrypt(encrypted)
    s = str(decrypted,'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)


    df['Libro'] = df['Libro'].astype(str)
    df['Autor'] = df['Autor'].astype(str)
    df['Editorial'] = df['Editorial'].astype(str)
    df['Proveedor'] = df['Proveedor'].astype(str)
    df['Materia'] = df['Materia'].astype(str)
    df['Coleccion'] = df['Coleccion'].astype(str)
    df['Libro'] = df['Libro'].str.strip()
    df['Autor'] = df['Autor'].str.strip()
    df['Editorial'] = df['Editorial'].str.strip()
    df['Proveedor'] = df['Proveedor'].str.strip()
    df['Materia'] = df['Materia'].str.strip()
    df['Coleccion'] = df['Coleccion'].str.strip()
    autores = sorted(df['Autor'].unique())
    libros = sorted(df['Libro'].unique())
    editoriales = sorted(df['Editorial'].unique())
    proveedores = sorted(df['Proveedor'].unique())
    materias =  sorted(df['Materia'].unique())
    colecciones  =  sorted(df['Coleccion'].unique())

    return df, {'autores': autores, 'libros':libros, 'editoriales':editoriales,
    'proveedores':proveedores, 'materias':materias, 'colecciones':colecciones
    }

@st.cache
def get_pool(data, labels, cat_f):
    return Pool(data, labels, cat_features=cat_f)


@st.cache
def predict(model, data):
    return model.predict(data)


@st.cache(suppress_st_warning=True,hash_funcs={catboost.core.Pool: lambda _: None})
def main_task():

    # Process
    if manual_entry:
        tmp_df = get_sample(df, 1)
    else:
        tmp_df = get_sample(df, num_prediccions)


    #tmp_df.reset_index(inplace=True)
    #st.text(tmp_df.head())
    res = process_data(tmp_df, cols_input, cols_empty, manual_entry)

    #test_real_labels = res['Pedido_real']
    test_real_labels = res.pop('Pedido_real')
    _ = res.pop('Pedido real')
    # Apply MinMaxScaler
    #scaler_data_ = np.load("./data/my_scaler.npy")
    #scaler_scale, scaler_min = scaler_data_[0], scaler_data_[1]

    #test_real_labels_scaled = test_real_labels * scaler_scale
    #test_real_labels_scaled += scaler_min


    #st.text("Real {}".format(tmp_df['Pedido real']))

    #res.drop(columns=['Tipo Articulo', 'Fecha'], inplace=True)
    #st.table(tmp_df)

    cat_features = np.where((res.dtypes != 'float32') & (res.dtypes != 'float64'))[0]
    #test_data = get_pool(res, test_real_labels_scaled, cat_features)
    #test_data = Pool(res, test_real_labels_scaled, cat_features=cat_features)
    test_data = Pool(res, test_real_labels, cat_features=cat_features)


    model = get_model()
    preds_scaled = model.predict(test_data)

    preds = preds_scaled

    #preds = preds_scaled - scaler_min
    #preds /= scaler_scale


    st.text("Predicción de pedidos: {}".format([int(np.round(p)) for p in preds]))
    st.text("Pedido real realizado: {}".format([p for p in test_real_labels]))



    resultados = pd.DataFrame(np.stack([[int(np.round(p)) for p in preds],np.array(test_real_labels)], axis=1), columns=['Predicciones','Pedidos reales'])
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=resultados.index, y=resultados['Pedidos reales'],
                        mode='lines+markers',
                        name='Pedidos reales'))
    fig.add_trace(go.Scatter(x=resultados.index, y=resultados['Predicciones'],
                        mode='lines+markers',
                        name='Predicciones'))

    st.plotly_chart(fig, use_container_width=True)
    shap_values = model.get_feature_importance(
        data=test_data,
        type='ShapValues',
        shap_calc_type='Approximate'
    )
    #sp_shape = shap_values.shape
    #st.text(shap_values)
    #st.text(shap_values.shape)
    #spv = shap_values.ravel()
    #st.text(spv)
    #st.text(spv.shape)
    #spv = spv - scaler_min
    #spv /= scaler_scale
    #st.text(spv)
    #st.text(spv.shape)
    #st.text(spv.reshape(sp_shape))
    #return test_data, res, shap_values.reshape(sp_shape), tmp_df, fig
    return test_data, res, shap_values, tmp_df, fig



column_defaults = {
        'Libro': str,
        'Tipo Articulo': str,
        'Editorial': str,
        'Coleccion': str,
        'Autor': str,
        'Proveedor': str,
        'Materia': str,
        'Idioma': str,
        'Mes 1     ': np.float32,
        'Mes 2     ': np.float32,
        'Mes 3     ': np.float32,
        'Mes 4     ': np.float32,
        'Mes 5     ': np.float32,
        'Mes 6     ': np.float32,
        'Mes 7     ': np.float32,
        'Mes 8     ': np.float32,
        'Mes 9     ': np.float32,
        'Mes 10    ': np.float32,
        'Mes 11    ': np.float32,
        'Mes 12    ': np.float32,
        'Mes 13    ': np.float32,
        'Mes 14    ': np.float32,
        'Mes 15    ': np.float32,
        'Mes 16    ': np.float32,
        'Mes 17    ': np.float32,
        'Mes 18    ': np.float32,
        'Mes 19    ': np.float32,
        'Mes 20    ': np.float32,
        'Mes 21    ': np.float32,
        'Mes 22    ': np.float32,
        'Mes 23    ': np.float32,
        'Mes 24    ': np.float32,
        'Stock': np.float32,
        'Ptes_Clientes': np.float32,
        'Ptes_Proveedor': np.float32,
        'Pedido real': int,
        'Date': str,
    }

cols_input = ['Mes 22','Mes 23','Mes 24','Proveedor','Materia','Coleccion',
    'Stock', 'Ptes_Clientes', 'Ptes_Proveedor', 'Pedido real']

cols_empty = [c for c in column_defaults.keys() if c not in cols_input]

mes_columns_dict = {}

for i in range(1,10):
    mes_columns_dict['Mes '+str(i)+'     '] = 'Mes '+str(i)
for i in range(10,25):
    mes_columns_dict['Mes '+str(i)+'    '] = 'Mes '+str(i)

mes_columns = []
acum_mes_columns = []
for i in range(1,25):
    mes_columns.append('Mes '+str(i))
    acum_mes_columns.append('Acum_Mes '+str(i+1))
acum_mes_columns = acum_mes_columns[:-1]

st.set_page_config(layout="wide", initial_sidebar_state='auto')


st.title("SIMULACIÓN PREDICCIÓN DE PEDIDOS DE LIBROS")
key = st.secrets["key"]



df, datos = load_data(key)
#st.text(df.head())

num_prediccions = st.sidebar.number_input("Número de predicciones (selección aleatoria de estos elementos de la base de datos)", min_value=10, max_value=len(df)-1, value=20)
manual_entry = st.sidebar.checkbox("Pincha para entrar manualmente los datos de la predicción a realizar", value=False)


st.sidebar.title(("Selecciona los parámetros del libro:"))

libro = st.sidebar.selectbox('Libro', datos['libros'])
autor = st.sidebar.selectbox('Autor', datos['autores'])
editorial = st.sidebar.selectbox('Editorial', datos['editoriales'])
proveedor = st.sidebar.selectbox('Proveedor', datos['proveedores'])
coleccion = st.sidebar.selectbox('Colección', datos['colecciones'])
materia = st.sidebar.selectbox('Materia', datos['materias'])

mes_24 = st.sidebar.number_input("Ventas mes 24", min_value=0)
mes_23 = st.sidebar.number_input("Ventas mes 23", min_value=0)
mes_22 = st.sidebar.number_input("Ventas mes 22", min_value=0)
ptes_clientes = st.sidebar.number_input("Pedidos ptes clientes", min_value=0)
ptes_proveedor = st.sidebar.number_input("Pedidos ptes proveedor", min_value=0)
stock = st.sidebar.number_input("Stock", min_value=0)
pedido_real = st.sidebar.number_input("Pedido real (para validación simulación)", min_value=0)


test_data, res, sp, tmp_df, fig = main_task()

st.table(tmp_df)
st.plotly_chart(fig, use_container_width=True)
expected_value = sp[0,-1]
shap_values = sp[:,:-1]
#shap_values = explainer.shap_values(X)
st.subheader("Visualizar importancia de cada variable para la predicción de cada libro")
object_sel = st.selectbox('Selecciona índice de objeto ', res.index.to_numpy())
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(expected_value, shap_values[object_sel,:], res.iloc[object_sel,:]))