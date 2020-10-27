# This is my first Dash coding
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly_express as px
import pandas as pd
from datetime import date
from dash.dependencies import Input, Output


#external_stylesheets = ['D:\vdata\SA\webdev\dash\index.css']

app = dash.Dash(__name__)

colors = {
    'background':'rgb(24, 24, 26)',
    'text':'white'}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    html.H1(id = 'Header',children='Topic Modeling',style={'textAlign':'center','color':colors['text'],'padding':'5px'}),

    html.Div(children='''
        This web UI is developed using Dash.
    ''',style={'textAlign':'center','color':colors['text'],'padding':'5px'}),

    html.Div(children='Knowledge Period',style={"font-size":20,'height':"500",'width':"auto",'backgroundColor':colors['background'],'color':colors['text'],'padding':'5px'}),
    html.Div(children='Please select a date',style={'height':"500",'width':"auto",'backgroundColor':colors['background'],'color':colors['text'],'padding':'5px'}),
    html.Div([dcc.DatePickerSingle(
        id='my-date-picker-single',
        date=date(2020, 10, 21),style={'padding':'5px'}
    ),
    html.Span(id='output-container-date-picker-single',style={'color':colors['text'],'textAlign':'center','padding':'8px'})]),
    html.Button('Show Topics',id='btn',style={'padding':'5px'}),

    html.Div(dcc.Graph(id='example-graph',figure=fig),style={'width':"860",'height':"500",'backgroundColor':colors['background'],'padding':'5px'}),

    
    html.Footer(children='This is a test web version of Topic Modelling. Copyright reserved.',
                style={'textAlign':'center','color':colors['text'],'backgroundColor':colors['background'],'padding':'5px'})
])

@app.callback(
    Output('output-container-date-picker-single', 'children'),
    [Input('my-date-picker-single', 'date')])
def update_output(date_value):
    string_prefix = 'You have selected: '
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        return string_prefix + date_string

if __name__ == '__main__':
    app.run_server(debug=False)
