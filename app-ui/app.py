import os
import sys
from pathlib import Path
import requests

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'app-ml', 'src'))
os.chdir(project_root)

# Force working directory to the project root
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from common.data_manager import DataManager
from common.utils import read_config, make_prediction_figures

# Load configuration using utils function
config_path = project_root / 'config' / 'config.yaml'
config = read_config(config_path)

# Override host for Docker environment if environment variable is set
inference_api_host = os.environ.get('INFERENCE_API_HOST', config.get('inference_api', {}).get('host', 'localhost'))
inference_api_port = config.get('inference_api', {}).get('port', 5001)
inference_api_endpoint = config.get('inference_api', {}).get('endpoint', '/run-inference')
INFERENCE_API_URL = f"http://{inference_api_host}:{inference_api_port}{inference_api_endpoint}"

# Initialize data manager and production database
data_manager = DataManager(config)
data_manager.initialize_prod_database()

# Use the default Bootstrap (light) theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the layout
app.layout = dbc.Container([
    dcc.Store(id='shared-xaxis-range'), # Store of data for zooming to the x-axis
    dcc.Store(id='inference-trigger', data=0),  # Store for inference trigger
    dbc.Row([
        # Control Panel
        dbc.Col([
            html.H4("Control Panel", style={"color": "#222"}),
            html.Div([
                html.Label("Plots display time (last N hours)", style={"color": "#222"}),
                dcc.Input(
                    id='lookback-hours',
                    type='number',
                    min=1,
                    step=1,
                    value=config['ui']['default_lookback_hours'],
                    style={"marginBottom": "16px", "width": "100%"}
                ),
                html.Label("Select features to display", style={"marginTop": "10px", "color": "#222"}),
                dcc.Dropdown(
                    id='parameter-dropdown',
                    options=[
                        {'label': 'Temperature', 'value': 'temp'},
                        {'label': 'Humidity', 'value': 'hum'},
                        {'label': 'Wind Speed', 'value': 'windspeed'},
                        {'label': 'Week Day', 'value': 'weekday'},
                        {'label': 'Working Day', 'value': 'workingday'},
                        {'label': 'Weather', 'value': 'weathersit'},
                    ],
                    value=['temp', 'hum'],
                    multi=True,
                    style={"backgroundColor": "#fff", "color": "#222"}
                ),
                html.Div([
                    dbc.Button(
                        "Predict Next Step", 
                        id="run-inference-btn", 
                        color="primary", 
                        n_clicks=0, 
                        style={"marginTop": "16px", "width": "100%", "marginBottom": "8px"}
                    ),
                    html.Div(id="inference-status", style={"fontSize": "12px", "color": "#666"})
                ]),
            ], style={"backgroundColor": "#fff", "borderRadius": "12px", "padding": "16px", "border": "1px solid #e0e0e0"}),
            html.Div([
                html.H5("ML Application Overview", style={"marginTop": "24px", "color": "#222"}),
                html.Ul([
                    html.Li("The ML application running an end-to-end ML pipeline (preprocessing, feature engineering, inference, postprocessing) in real-time"),
                    html.Li("The top plot shows predicted bike count for the next hour vs true values."),
                    html.Li("The bottom plot displays selected features associated with the predicted bike count."),
                    html.Li("Plots update when 'Predict Next Step' is clicked."),
                    html.Li("The UI app and inference pipeline run in 2 Docker containers."),
                    html.Li("The data and model are stored and shared in Docker volumes."),
                ], className="overview-list", style={"color": "#444", "fontSize": "15px", "padding": "10px", "lineHeight": "1.4"})
            ], style={"backgroundColor": "#fff", "borderRadius": "12px", "padding": "20px", "border": "1px solid #e0e0e0",
                      "height": "100%", "display": "flex", "flexDirection": "column", "marginBottom": "10px", "marginTop": "10px"
                          }
                          )
        ], width=3, style={"display": "flex", "flexDirection": "column", "height": "100%", "paddingTop": "10px"}),
        # Graphs
        dbc.Col([
            html.H5("Real-time Bike Count Predictions", style={"marginBottom": "1px", "color": "#222"}),
            dcc.Graph(id='graph-1', clear_on_unhover=True, style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "8px",
                "height": "50%", 
                "width": "100%",
                "minHeight": 0   
            }),
            html.H5("Features Data", style={"marginBottom": "1px", "color": "#222"}),
            dcc.Graph(id='graph-2', clear_on_unhover=True, style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "8px",
                "height": "50%", 
                "width": "100%",
                "minHeight": 0
            }),
        ], width=9, style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "padding": "0",
            "gap": "10px",
            "paddingBottom": "10px",
            "paddingTop": "10px",
            "paddingRight": "10px",
        })
            ], align="stretch", style={"flex": 1, "height": "100%"})
        ], fluid=True, style={"height": "100vh", "minHeight": "100vh", "backgroundColor": "#e9e9f0"})


# Callback to update the shared x-axis range when either plot is zoomed or panned
@callback(
    Output('shared-xaxis-range', 'data'),
    [Input('graph-1', 'relayoutData'),
     Input('graph-2', 'relayoutData')],
    [State('shared-xaxis-range', 'data')]
)
def sync_xaxis_range(relayout1, relayout2, stored_range):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    relayout = relayout1 if trigger == 'graph-1' else relayout2 if trigger == 'graph-2' else None
    if relayout and ('xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout):
        return [relayout['xaxis.range[0]'], relayout['xaxis.range[1]']]
    elif relayout and 'xaxis.autorange' in relayout:
        return None  # Reset to autorange
    return stored_range


# Main callback to update both plots, now using the shared x-axis range
@callback(
    [Output('graph-1', 'figure'),
     Output('graph-2', 'figure')],
    [
     Input('lookback-hours', 'value'),
     Input('parameter-dropdown', 'value'),
     Input('shared-xaxis-range', 'data'),
     Input('inference-trigger', 'data')]  # Use inference trigger instead of button clicks
)
def update_graphs(lookback_hours, parameters, shared_xrange, inference_trigger):
    try:
        if lookback_hours is None or lookback_hours < 1:
            lookback_hours = config['ui']['default_lookback_hours']
        try:
            df_pred = data_manager.load_prediction_data()
        except Exception:
            df_pred = None
        df_prod = data_manager.load_prod_data()
        fig1, fig2 = make_prediction_figures(
            df_prod, df_pred, parameters, config, lookback_hours, shared_xrange
        )
        return fig1, fig2
    except Exception as e:
        fig1 = go.Figure()
        fig2 = go.Figure()
        for fig in [fig1, fig2]:
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=40, r=40, t=20, b=20),
                height=330,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Value",
                annotations=[{
                    'text': f'Error loading data: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 20}
                }],
                plot_bgcolor='#fff',
                paper_bgcolor='#fff'
            )
        return fig1, fig2


# Callback for the inference button
@callback(
    Output('inference-status', 'children'),
    Output('run-inference-btn', 'disabled'),
    Output('inference-trigger', 'data'),  # Add a trigger for plot updates
    Input('run-inference-btn', 'n_clicks'),
    prevent_initial_call=True
)
def trigger_inference(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "", False, 0
    try:
        # Disable button during inference
        # Call the inference API in the other container
        response = requests.post(INFERENCE_API_URL)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return f"✅ Prediction completed for {result.get('timestamp', '')}", False, n_clicks
            else:
                return f"Error: {result.get('message', 'Unknown error')}", False, 0
        else:
            return f"Error: {response.text}", False, 0
    except Exception as e:
        return f"Error: {str(e)}", False, 0


server = app.server

if __name__ == '__main__':
    # Use debug=True for development, debug=False for production
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)