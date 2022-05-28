from matplotlib.pyplot import figure
import csv
import pandas as pd
import mne
import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
#import plotly.graph_objects as go
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from sqlalchemy import false
from sympy import N
from torch import _register_device_module
#global var that remembers previous montage type
setting = ["common_reference"]
#global var that remembers the previous click data
click_start_setting = [0]
#stores all notes from datatable + text area
notes = [] # when writing the save function make sure to add in labels from notesData[] because it will be empty
for i in range(0,50):
    notes.append({'start' : 10* i, 'end' : 10 * i + 10, 'label' : '', 'comments' : ''})
#import eeg file
raw_eeg_filepath = r"C:\Users\mzeng\Desktop\mne-dash-master\data\00010736_s001_t001.edf"
raw_data = mne.io.read_raw_edf(raw_eeg_filepath, verbose=True).crop(tmin=0, tmax=500).resample(250)#note: graphs only display first 500s bc of tmax

#import x and y axis for psd graph from eeg file
psds_axis, freq_axis = mne.time_frequency.psd_welch(raw_data)#, 250, fmax=50.0, n_per_seg=150, average='mean', verbose=False)
psds_axis = 10 * np.log10(psds_axis)

#import x and y axis for raw plot from eeg file and downsample to 64Hz
resample_freq = 64
raw_data = raw_data.resample(sfreq = resample_freq)
voltage_axis, time_axis = raw_data.get_data(return_times = True, tmin = 0, tmax = 500)
#epochs = mne.io.EpochsArray()#TRYING TO CREATE EPOCHS ARRAY FROM RAW DATA RIGHT NOW USING NATMEG.SE LINK
n_epochs = int((time_axis.shape[0] / resample_freq) / 10)

#import available channel names into checklist(remove last 4 channels bc unimportant)
options_list = []
ch_names = raw_data.info['ch_names']
ch_names = ch_names[:-4] 
for ch_name in ch_names:
    options_list.append({'label': ch_name,'value': ch_name})

#set default channel_checklist values to on(for select all switch)
value_list = []
for ch_name in ch_names:
     value_list.append(ch_name)
#list of option names for the bipolar checklist
bipolar_options = [
    {'label': 'FP1-F3','value': 'FP1-F3'},
    {'label': 'F3-C3','value': 'F3-C3'},
    {'label': 'C3-P3','value': 'C3-P3'},
    {'label': 'P3-O1','value': 'P3-O1'},

    {'label': 'FP2-F4','value': 'FP2-F4'},
    {'label': 'F4-C4','value': 'F4-C4'},
    {'label': 'C4-P4','value': 'C4-P4'},
    {'label': 'P4-O2','value': 'P4-O2'},

  
    {'label': 'FP1-F7','value': 'FP1-F7'},
    {'label': 'F7-T3','value': 'F7-T3'},
    {'label': 'T3-T5','value': 'T3-T5'},
    {'label': 'T5-O1','value': 'T5-O1'},

    {'label': 'FP2-F8','value': 'FP2-F8'},
    {'label': 'F8-T4','value': 'F8-T4'},
    {'label': 'T6-O2','value': 'T6-O2'},
    {'label': 'T4-T6','value': 'T4-T6'},

    {'label': 'FZ-CZ','value': 'FZ-CZ'},
    {'label': 'CZ-PZ','value': 'CZ-PZ'},
]
#adding all bipolar channel names to list
bipolar_ch_names=[]
for item in bipolar_options:
    bipolar_ch_names.append(item.get('label'))
#hard coding all voltage axes
bipolar_voltages = [
    {'label': 'FP1-F3','voltage': voltage_axis[0] - voltage_axis[2]},
    {'label': 'F3-C3','voltage': voltage_axis[2] - voltage_axis[4]},
    {'label': 'C3-P3','voltage': voltage_axis[4] - voltage_axis[6]},
    {'label': 'P3-O1','voltage': voltage_axis[6] - voltage_axis[8]},

    {'label': 'FP2-F4','voltage': voltage_axis[1] - voltage_axis[3]},
    {'label': 'F4-C4','voltage': voltage_axis[3] - voltage_axis[5]},
    {'label': 'C4-P4','voltage': voltage_axis[5] - voltage_axis[7]},
    {'label': 'P4-O2','voltage': voltage_axis[7] - voltage_axis[9]},

    {'label': 'FP1-F7','voltage': voltage_axis[0] - voltage_axis[10]},
    {'label': 'F7-T3','voltage': voltage_axis[10] - voltage_axis[12]},
    {'label': 'T3-T5','voltage': voltage_axis[12] - voltage_axis[14]},
    {'label': 'T5-O1','voltage': voltage_axis[14] - voltage_axis[8]},

    {'label': 'FP2-F8','voltage': voltage_axis[1] - voltage_axis[11]},
    {'label': 'F8-T4','voltage': voltage_axis[11] - voltage_axis[13]},
    {'label': 'T6-O2','voltage': voltage_axis[13] - voltage_axis[9]},
    {'label': 'T4-T6','voltage': voltage_axis[9] - voltage_axis[15]},

    {'label': 'FZ-CZ','voltage': voltage_axis[18] - voltage_axis[19]},
    {'label': 'CZ-PZ','voltage': voltage_axis[19] - voltage_axis[20]},
]
#use bipolar_voltages to create a clean volage_axis
bipolar_voltage_axis = []
for item in bipolar_voltages:
    bipolar_voltage_axis.append(item.get('voltage'))
#tab styling
tabs_styles = {
    'height': '40px',
    'width': '250px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'backgroundColor': '#EBF2FF'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}


#################################################################################
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            html.Div(className ='eight columns',#tabs
                children=[
                    dbc.Tabs(
                        children=[
                            dbc.Tab(label='Raw Data', tab_id='raw_graph'),
                            dbc.Tab(label='PSD Graph', tab_id='psd_graph'),
                        ],
                        active_tab='raw_graph', 
                        #vertical = True,
                        id="current_tab"
                    )
                ]
            ),
            width={"size": 2, "order": "first", "offset": 0}
        ),
        dbc.Col(
            html.Div(className ='one columns',#dropdown
                children=[
                    dbc.DropdownMenu(
                        label="Montage type",
                        children=[
                            dbc.DropdownMenuItem("Common Reference", n_clicks=1, id="common_reference",style = tabs_styles),
                            dbc.DropdownMenuItem("Bipolar", n_clicks=0, id="bipolar", style = tabs_styles),
                        ],
                        id='active_dropdown'
                    )
                ]
            ),
            width={"size": 1, "order": "second", "offset": 0}
        ),
        dbc.Col(
            html.Div(className ='one columns',#save button
            children=[
                html.Button('Save', id='save_button', n_clicks=0),
                dcc.Download(id="download_csv")
            ]),
        width={"size": 1, "order": "last", "offset": 8}
        )
    ]),
    
    dbc.Row([
        dbc.Col(
            html.Div(className ='twelve columns',
                children=[
                    html.P('Channels'),
                    dbc.Checklist(#select all switch
                        options=[
                            {'label': 'Select All', 'value': 1},
                        ],
                        value=[1],
                        switch = True,
                        id = 'select_checklist',
                    ),
                    dbc.Checklist(#channel checklist
                        options=options_list,
                        value = value_list,
                        switch = True,
                        id = 'channel_checklist',
                    ),
                    html.Br(),
                    html.P("Enter a value between " + "1" + "-" + str(n_epochs)),
                    dbc.Input(#epoch input text box
                        placeholder='Enter a value...',
                        type="number",
                        value=1,
                        min=1, max=n_epochs, step=1,
                        id = 'epoch_input',
                        #style = {'color' : 'white'}
                    )
                ]
            ),
        width={"size": 1, "order": "first", "offset": 0}
        ),
        dbc.Col(
            html.Div(className ='twelve columns',#main graph
                children=[
                #app callback returns main graph here
                dcc.Graph(
                    figure={},
                    style={'height': 800},
                    id = 'main_graph'
                ),                
            ],
            id ='main_div'
            ),
        width={"size": 10, "order": "second", "offset": 0}
        ),
        html.Br(),
    ]),
    html.Br(),
    dbc.Row([
         dbc.Col(
            html.Div(className ='twelve columns',#epoch graph
                    children=[#returns epoch div here
                        dcc.Graph(
                            figure={},
                            style={'height': 400},
                            id = 'epoch_graph'
                        )
                    ],
            id ='epoch_div'
            ),
        width={'size': 8, 'offset' : 1}
        ),
        dbc.Col(
            html.Div(className ='twelve columns',
                children=[
                    dbc.Row([
                        dbc.Col(
                            dbc.RadioItems(#label checklist
                                options=[
                                    {'label': 'W','value': 'W'},
                                    {'label': 'N1','value': 'N1'},
                                    {'label': 'N2','value': 'N2'},
                                    {'label': 'N3','value': 'N3'},   
                                    {'label': 'REM','value': 'REM'},   
                                ],
                                value = 'W',
                                id = 'eeg_type_checklist',
                            ),
                        width={"size": 2, "order": "first"}
                        ),
                        dbc.Col(
                            html.Div(className ='twelve columns',
                                children=[
                                    dash_table.DataTable(#data table
                                        id='notes_table',
                                        columns=([
                                            {'id': 'Start_notes', 'name': 'Start(s)'},
                                            {'id': 'End_notes', 'name': 'End(s)'},
                                            {'id': 'Label', 'name': 'Label'},
                                            ]
                                        ),
                                        data=[
                                            dict(Start_notes=10 * i, End_notes = 10 * i + 10, Label = "")
                                            for i in range(0, 50)
                                        ],                                    
                                    ),
                                ],
                            ),
                        style={'height' : 400, 'overflowY' : 'scroll'},
                        width={"size": 5, 'order' : 'second'}
                        ),
                        dbc.Col(
                            html.Div(className ='twelve columns',
                                children=[
                                    dcc.Textarea(#text box
                                        id='textarea',
                                        value='',
                                        style={'width': '100%', 'height': 400, 'overflowY': 'scroll', 'color' : 'white'},
                                    ),
                                ],
                            ),
                        width={"size": 5, "order": "last"}
                        )
                    ]),
                ],
            id = 'notes_div'    
            ),
        width={"size": 3, "order": "third"}
        )
    ]),
    html.P(id='placeholder'),
])

@app.callback(
    Output("download_csv", "data"),
    Input("save_button", "n_clicks"),
    State('notes_table', 'data'),
    prevent_initial_call=True,
)
def download(click, notesData):
    for i in range(0,50):
        notes[i]['label'] = notesData[i]['Label']
    
   
    df = pd.DataFrame.from_records(notes)

    return dcc.send_data_frame(df.to_csv, "notes.csv", index= False)


# @app.callback(#function made in case there was an error first time because click_start_setting[0] = 0 at launch
#but after commenting it out nothing broke
# 	Output('placeholder', 'style'),
#     Input('main_graph', 'clickData'),
# )
# def click_start_setting_update(clickData, label, notesData):
#     ctx = dash.callback_context
#     if clickData == None:
#         raise PreventUpdate

#     click_time_raw = clickData["points"][0]['x']
#     click_start = click_time_raw - (click_time_raw % 10)
#     click_start_setting[0] = click_start

#     return {}


@app.callback(
    Output('textarea', 'value'),
    Input('main_graph', 'clickData'),
    State('textarea', 'value'),
)

def textBox_update(clickData, textState):
    print('running')
    ctx = dash.callback_context
    if clickData == None:
        raise PreventUpdate

    #first save the previous state
    notes[int(click_start_setting[0]/10)]['comments'] = textState
    #update click_start_setting to hold new clickData
    click_time_raw = clickData["points"][0]['x']
    click_start = click_time_raw - (click_time_raw % 10)
    click_start_setting[0] = click_start

    #return text corresponding to new clickData
    #print(notes)
    return notes[int(click_start_setting[0]/10)]['comments']
    

@app.callback(
	Output('notes_table', 'data'), 
    Input('eeg_type_checklist', 'value'),
    State('main_graph', 'clickData'),
    State('notes_table', 'data'),
)
#finds the time where you clicked and updates the "Label" column to the label(W,N0,N1,N2,N3,REM) that you picked
def table_update(label, clickData, notesData):
    ctx = dash.callback_context
    if clickData == None:
        raise PreventUpdate
    #print(clickData)
    click_time_raw = clickData["points"][0]['x']
    click_start = click_time_raw - (click_time_raw % 10)

    notesData[int(click_start/10)]['Label'] = label

    return notesData



@app.callback(
	Output('channel_checklist', 'options'),
    Input('common_reference', 'n_clicks'),
    Input('bipolar', 'n_clicks')
)
def montage_type(common_ref_clicks, bipolar_clicks):#these args do nothing(they only stop the error)
    #set button_id to current selected in dropdown
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "common_reference"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #print(button_id)
    if button_id == 'common_reference':
        select = options_list
        return select;
    else:#if button_id == 'bipolar' 
        #set new bipolar checklist all hardcoded
        select = bipolar_options
        return select;

    

#select/deselect all switch function
@app.callback(
	Output('channel_checklist', 'value'),
	Input('select_checklist', 'value'),
    Input('common_reference', 'n_clicks'),
    Input('bipolar', 'n_clicks')
)
def select_toggle(select_all, common_ref_clicks, bipolar_clicks):
    #set button_id to current selected in dropdown
    #print("select_toggle used")
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "common_reference"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
   
    if button_id == "common_reference":
        setting[0] = "common_reference"
    elif button_id == "bipolar":
        setting[0] = "bipolar"
    
    # print("select_all: ", select_all)
    # print("setting[0]: ", setting[0])
    # print("ctx.triggered[0]: ", ctx.triggered[0])
    # print(ctx.triggered[0]["prop_id"].split(".")[0])
    #print(select_all)

    if len(select_all) == 0:
        deselect  = []
        return deselect
    else:
        select = []
        if setting[0] == 'common_reference':
            for ch_name in ch_names:
                select.append(ch_name)
            return select
        else:#if setting[0] == 'bipolar'
            bipolar_values=[]
            for item in bipolar_options:
                bipolar_values.append(item.get('value'))
            return bipolar_values
            
#updates main graph based on changes to channel switches and current tab
@app.callback(
	Output('main_graph', 'figure'),
    Output('epoch_div', 'children'),
	Input('channel_checklist', 'value'),
    Input('current_tab', 'active_tab'),
    Input('epoch_input', 'value'),
    Input("common_reference", "n_clicks"),
    Input("bipolar", "n_clicks"),
    Input('main_graph', 'clickData'),
    #State('main_graph', 'figure')
)
def graph_mod(selectedChannels, active_tab, epoch_input, common_ref_clicks, bipolar_clicks, clickData):
    #################################################################set state of setting[0]
    ctx = dash.callback_context
    # print(ctx.triggered[0]["prop_id"].split(".")[1])
    button_id = ""
    if not ctx.triggered:
        button_id = "common_reference"
    else:
        if ctx.triggered[0]["prop_id"].split(".")[1] == 'n_clicks':
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        else:
            button_id = button_id
    
    if button_id == "common_reference":
        setting[0] = "common_reference"
    elif button_id == "bipolar":
        setting[0] = "bipolar"
#############################################################update epoch graph based on channel switches and current tab and clickData
    #print(ctx.triggered[0])
    if clickData != None and ctx.triggered[0]["prop_id"] == 'main_graph.clickData':
        click_time_raw = clickData["points"][0]['x']
        click_end = click_time_raw - (click_time_raw % 10) + 10
        epoch_input = click_end/10
#############################################################
    n_channels = len(selectedChannels)
    #create empty subplots
    if n_channels == 0:
        return [{},{}]
    elif(epoch_input == None or epoch_input < 1 or epoch_input > 50):#return null if no channels selected
        raise PreventUpdate
    else:
        yaxis_step = 1. / n_channels

        epoch_subplots = make_subplots(
        rows=n_channels, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing= 0.00,
        subplot_titles= selectedChannels
        )
    time_init = (epoch_input - 1) * 10
    time_final = epoch_input * 10 + 0.1
    start, stop = raw_data.time_as_index([time_init, time_final])
    epoch_voltage_axis, epoch_time_axis = raw_data[:len(ch_names), start:stop]
    #print(epoch_time_axis)

#hard coding all voltage axes to create epoch_bipolar_voltage_axis
    epoch_bipolar_voltages = [
        {'label': 'FP1-F3','voltage': epoch_voltage_axis[0] - epoch_voltage_axis[2]},
        {'label': 'F3-C3','voltage': epoch_voltage_axis[2] - epoch_voltage_axis[4]},
        {'label': 'C3-P3','voltage': epoch_voltage_axis[4] - epoch_voltage_axis[6]},
        {'label': 'P3-O1','voltage': epoch_voltage_axis[6] - epoch_voltage_axis[8]},

        {'label': 'FP2-F4','voltage': epoch_voltage_axis[1] - epoch_voltage_axis[3]},
        {'label': 'F4-C4','voltage': epoch_voltage_axis[3] - epoch_voltage_axis[5]},
        {'label': 'C4-P4','voltage': epoch_voltage_axis[5] - epoch_voltage_axis[7]},
        {'label': 'P4-O2','voltage': epoch_voltage_axis[7] - epoch_voltage_axis[9]},

        {'label': 'FP1-F7','voltage': epoch_voltage_axis[0] - epoch_voltage_axis[10]},
        {'label': 'F7-T3','voltage': epoch_voltage_axis[10] - epoch_voltage_axis[12]},
        {'label': 'T3-T5','voltage': epoch_voltage_axis[12] - epoch_voltage_axis[14]},
        {'label': 'T5-O1','voltage': epoch_voltage_axis[14] - epoch_voltage_axis[8]},

        {'label': 'FP2-F8','voltage': epoch_voltage_axis[1] - epoch_voltage_axis[11]},
        {'label': 'F8-T4','voltage': epoch_voltage_axis[11] - epoch_voltage_axis[13]},
        {'label': 'T6-O2','voltage': epoch_voltage_axis[13] - epoch_voltage_axis[9]},
        {'label': 'T4-T6','voltage': epoch_voltage_axis[9] - epoch_voltage_axis[15]},

        {'label': 'FZ-CZ','voltage': epoch_voltage_axis[18] - epoch_voltage_axis[19]},
        {'label': 'CZ-PZ','voltage': epoch_voltage_axis[19] - epoch_voltage_axis[20]},
    ]
    epoch_bipolar_voltage_axis=[]
    for item in epoch_bipolar_voltages:
        epoch_bipolar_voltage_axis.append(item.get('voltage'))

    #fill each subplot with raw data corresponding to the channel name
    #print(button_id)
    if setting[0] == 'bipolar':
        for selected_idx, selected in enumerate(selectedChannels, start = 1):
            epoch_subplots.add_trace(
                go.Scatter(
                    x=epoch_time_axis, 
                    y=epoch_bipolar_voltage_axis[bipolar_ch_names.index(selected)],
                    mode='lines', 
                    name=selected, 
                    line=go.scatter.Line(color="black")
                ),
                row = selected_idx,
                col = 1
            )
    else:
        for selected_idx, selected in enumerate(selectedChannels, start = 1):
            epoch_subplots.add_trace(
                go.Scatter(
                    x=epoch_time_axis, 
                    y=epoch_voltage_axis[ch_names.index(selected)],
                    mode='lines', 
                    name=selected, 
                    line=go.scatter.Line(color="black")
                ),
                row = selected_idx,
                col = 1
            )

    epoch_div_child = dcc.Graph(
        figure=epoch_subplots,
        style={'height': 400},
        id = 'epoch_graph'
     )

    #create titles(annotations)
    annotations = [
        Annotation(
            x=-0.05, y=0, 
            xref='paper', yref='y%d' % (selected_idx + 1),
            text=selected, 
            font=Font(size=15),
            showarrow=False
        )
        for selected_idx, selected in enumerate(selectedChannels)
    ]

    #update layout for annotations to actually appear
    epoch_subplots.update_layout(
        # width=, 
        # height=,
        paper_bgcolor='rgba(255,255,255, 1.0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        margin=dict(l=150, r=10, t=0, b=0),
        yaxis=dict(domain=[1 - yaxis_step, 1]),
        annotations=annotations,
        showlegend=False
    )

    ##############################################################################
    #psd graph tab
    if active_tab == 'psd_graph':
        #add all selected psd graphs to psd_data_list under dcc.graph
        psd_data_list = []

        for selected in selectedChannels:
            psd_data_format = dict(x = freq_axis, y = psds_axis[ch_names.index(selected)], name = selected, marker = dict(color=px.colors.qualitative.Dark24))
            psd_data_list.append(psd_data_format)  

        #use psd_data_list to create figure being returned
        new_psd_graph = dict(
            data = psd_data_list,
            layout=dict(
                title='PSD',
                xaxis = dict(title = 'Frequency (Hz)'),
                yaxis = dict(title = 'Amplitude'),
                showlegend=True,
                legend=dict(
                    x=100,
                    y=1.0
                ),
                margin=dict(l=80, r=80, t=80, b=80)
            )
        )

        return [new_psd_graph, epoch_div_child]

    #raw plot tab
    elif active_tab == 'raw_graph':
        yaxis_step = 1. / n_channels
        
        #create empty subplots
        if(n_channels == 0):#return null if no channels selected
            return {}
        else:
            raw_subplots = make_subplots(
            rows=n_channels, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing= 0.00,
            subplot_titles= selectedChannels
            )
            
        
        #fill each subplot with raw data corresponding to the channel name
        if setting[0] == 'bipolar':
           for selected_idx, selected in enumerate(selectedChannels, start = 1):
                raw_subplots.add_trace(
                    go.Scatter(
                        x=time_axis, 
                        y=bipolar_voltage_axis[bipolar_ch_names.index(selected)],
                        mode='lines', 
                        name=selected, 
                        line=go.scatter.Line(color="black")
                    ),
                    row = selected_idx,
                    col = 1
                )  
        else:
            for selected_idx, selected in enumerate(selectedChannels, start = 1):
                raw_subplots.add_trace(
                    go.Scatter(
                        x=time_axis, 
                        y=voltage_axis[ch_names.index(selected)],
                        mode='lines', 
                        name=selected, 
                        line=go.scatter.Line(color="black")
                    ),
                    row = selected_idx,
                    col = 1
                )
            #Commented code is attempt to display raw subplots using low level plotly interface instead of go objects high level interface
            # raw_data_list = []
            # raw_data_format = dict(type = "bar", x = time_axis, y = voltage_axis[selected_idx-1], name = selected, marker = dict(color=px.colors.qualitative.Dark24))
            # raw_data_list.append(raw_data_format)
            # dict(
            #         data= raw_data_list,
            #         layout=dict(
            #             title='Raw Plot',
            #             xaxis = dict(title = 'Time'),
            #             yaxis = dict(title = 'Volts'),
            #             showlegend=True,
            #             legend=dict(
            #                 x=100,
            #                 y=1.0
            #             ),
            #             margin=dict(l=80, r=80, t=80, b=80)
            #         )
            #     )

        #copy raw_subplots to a figure called new_raw_data_graph
        new_raw_data_graph = raw_subplots
        
        #create titles(annotations)
        annotations = [
            Annotation(
                x=-0.05, y=0, 
                xref='paper', yref='y%d' % (selected_idx + 1),
                text=selected, 
                font=Font(size=15),
                showarrow=False
            )
            for selected_idx, selected in enumerate(selectedChannels)
        ]

        #update layout for annotations to actually appear
        raw_subplots.update_layout(
            # width=, 
            # height=,
            paper_bgcolor='rgba(255,255,255, 1.0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            margin=dict(l=150, r=0, t=0, b=0),
            yaxis=dict(domain=[1 - yaxis_step, 1]),
            annotations=annotations,
            showlegend=False
        )

        raw_subplots.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.1, rangeslider_range=[0, .001],
                        col=1, row=n_channels)

        return [new_raw_data_graph, epoch_div_child]


if __name__ == '__main__':
    app.run_server(debug=True)
