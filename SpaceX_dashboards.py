import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, callback, Input, Output
from plotly import express as px


URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
df = pd.read_csv(URL)
min_payload = df['Payload Mass (kg)'].min()
max_payload = df['Payload Mass (kg)'].max()

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Space X launch Records Dashboards'),

    dcc.Dropdown(
        id='site-dropdown',
        options=
            [{'label': 'All Sites', 'value': 'ALL'},
             {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
             {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
             {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
             {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}
             ],
        value='ALL',
        placeholder='Select a Launch Site here',
        searchable=True   #attribute to be True so we can enter keywords to search launch sites
    ),

    html.Div(id='success-pie-chart'),

    dcc.RangeSlider(id='payload-slider',
                    min=0, max=10000, step=1000,
                    marks={0: '0',
                           2500: '2500',
                           5000: '5000',
                           7500: '7500',
                           10000: '10000'},
                    value=[min_payload, max_payload]
    ),
    html.Br(),
    html.Div(id='success-payload-scatter-chart')
])


@callback(Output(component_id='success-pie-chart', component_property='children'),
          Input(component_id='site-dropdown', component_property='value'))
def viz(choosen_site):
    if choosen_site == 'ALL':
        fig = px.pie(df, values='class', names='Launch Site')
        text = f'Total Success Launches By Site'
        return html.H2(text), dcc.Graph(figure=fig)
    elif choosen_site:
        df_site = df[df['Launch Site'] == choosen_site]
        df_for_pie = df_site[['class', 'Launch Site']].groupby('class').size().reset_index()
        df_for_pie.columns = ['names', 'class_percents']
        df_for_pie['names'] = ['unsuccess', 'success']
        fig = px.pie(df_for_pie, values='class_percents', names='names')
        text = f'Total Success Launches For Site {choosen_site}'
        return html.H2(text), dcc.Graph(figure=fig)

@callback(Output(component_id='success-payload-scatter-chart', component_property='children'),
          [Input(component_id='site-dropdown', component_property='value'),
           Input(component_id="payload-slider", component_property="value")])
def slider_output(choosen_site, slider):
    df_slider = df[(df['Payload Mass (kg)'] > slider[0]) & (df['Payload Mass (kg)'] < slider[1])]
    if choosen_site == 'ALL':
        fig = px.scatter(df_slider, x=df_slider['Payload Mass (kg)'], y=df_slider['class'], color="Booster Version Category",)

        return dcc.Graph(figure=fig)
    elif choosen_site:
        df_site = df_slider[df_slider['Launch Site'] == choosen_site]
        fig = px.scatter(df_site, x=df_site['Payload Mass (kg)'], y=df_site['class'], color="Booster Version Category")
        return dcc.Graph(figure=fig)



if __name__ == '__main__':
    app.run_server()