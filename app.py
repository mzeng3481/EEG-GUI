import importlib
import uuid
from pathlib import Path
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from scipy import stats
import pandas as pd
import numpy as np
import mne

import plotly.express as px
from plotly import tools
# import plotly.graph_objs as go
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# cool components
import dash_uploader as du
import visdcc
import dash_daq as daq
import dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash_extensions import Download

from model.eeg_pipeline import standardize_sensors, downsample, highpass, remove_line_noise, get_brain_waves_power

import torch
from model.EEGGraphDataset import EEGGraphDataset
from model.shallow_EEGGraphConvNet import EEGGraphConvNet
from torch_geometric.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

app = dash.Dash(
	__name__,
	suppress_callback_exceptions=True,
	meta_tags=[
		{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
	],
	external_stylesheets=[
		"https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css",
		# "https://codepen.io/chriddyp/pen/bWLwgP.css",
		# "https://codepen.io/rmarren1/pen/mLqGRg.css",
		dbc.themes.BOOTSTRAP
		],
)
server = app.server
# NOTE: use_upload_id=True for saving files with session ID
UPLOAD_FOLDER_ROOT = r"Dash_Uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=True)

def display_psd_graph():

	X_psd = np.load("./data/X_psd.npy", allow_pickle=True)
	n_windows = X_psd.shape[0]
	X_psd_reshaped = X_psd.reshape((n_windows, 8, 6))
	X_psd_reshaped_avg = np.mean(X_psd_reshaped, axis=1)

	X_psd_reshaped_avg.T[0, :] = stats.zscore(X_psd_reshaped_avg.T[0, :])
	X_psd_reshaped_avg.T[1, :] = stats.zscore(X_psd_reshaped_avg.T[1, :])
	X_psd_reshaped_avg.T[2, :] = stats.zscore(X_psd_reshaped_avg.T[2, :])
	X_psd_reshaped_avg.T[3, :] = stats.zscore(X_psd_reshaped_avg.T[3, :])
	X_psd_reshaped_avg.T[4, :] = stats.zscore(X_psd_reshaped_avg.T[4, :])
	X_psd_reshaped_avg.T[5, :] = stats.zscore(X_psd_reshaped_avg.T[5, :])

	fig = px.imshow(X_psd_reshaped_avg.T, 
					labels=dict(x="Time (s)", y="Brain Rhythms", color="PSD z-score"),
					x=np.array(range(X_psd_reshaped_avg.T.shape[1])) * 10,
					y=[r"$\delta$", r'$\theta$', r'$\alpha$', r'lower $\beta$', r'higher $\beta$', r'$\gamma$'],
					color_continuous_scale='RdBu_r', 
					aspect='equal')
	fig.update_xaxes(side="top")
	return fig

app.layout = html.Div(
	children=[
		# div for top banner and logos
		html.Div(
			className="banner",
			children=[
				html.Div(
					className="container scalable",
					children=[
						html.H2(
							id="banner-title",
							children=[
								html.A(
									"EEG Analytics",
									href="",
									style={
										"text-decoration": "none",
										"color": "inherit",
									},
								)
							],
						),
						html.A(
							id="",
							children=[
								html.Img(src=app.get_asset_url("mc_logo_black_white.jpg"))
							],
						),
						html.A(
							id="",
							children=[
								html.Img(src=app.get_asset_url("bioe_pic.jpg"))
							],
						),
					],
				)
			],
		),

		# div for all content below the banner
		html.Div(
			className='twelve columns model-options',
			children=[
				# div for sidebar
				html.Div(
					# id = "container scalable",
					className='one column',
					children = [
						daq.BooleanSwitch(
							id='trace-switch',
							on=False,
							color="#E84C26",
							label="Trace Plot",
							labelPosition="top"
						),						
						daq.BooleanSwitch(
							id='preprocessing-switch',
							on=False,
							color="#E84C26",
							label="Preprocess EEG",
							labelPosition="top"
						),
						daq.BooleanSwitch(
							id='eeg-gcnn-switch',
							on=False,
							color="#E84C26",
							label="EEG-GCNN",
							labelPosition="top"
						),
						daq.BooleanSwitch(
							id='interpret-switch',
							on=False,
							color="#E84C26",
							label="EEG-INTERPRET",
							labelPosition="top"
						),
						# dbc.Button("Open modal", id="open"),
						dbc.Modal(
							[
								dbc.ModalHeader("Preprocessing Options"),
								dbc.ModalBody([
									dcc.Checklist(
										id="preprocess-checklist",
										options=[
											{'label': 'Resample to 250Hz', 'value': 'RESAMPLE'},
											{'label': 'Highpass Filter at 1Hz', 'value': 'HIGHPASS'},
											{'label': 'Notch Filter at 50Hz', 'value': 'NOTCH'}
										],
										value=['RESAMPLE', 'HIGHPASS', 'NOTCH']
									),
									]
								),
								dbc.ModalFooter(
									dbc.Button("Apply", id="preprocessing-close", className="ml-auto")
								),
							],
							id="preprocessing-modal",
						),						
						dbc.Modal(
							[
								dbc.ModalHeader("Sensors Selection"),
								dbc.ModalBody([
									dcc.Checklist(
										id="sensors-checklist",
										options=[
											# {'label': 'S1', 'value': 'F1'},
											# {'label': 'S2', 'value': 'F2'},
											# {'label': 'S3', 'value': 'F3'}
										],
										value=[] #'F1', 'F2', 'F3']
									),
									]
								),
								dbc.ModalFooter(
									dbc.Button("Apply", id="trace-close", className="ml-auto")
								),
							],
							id="trace-modal",
						),
						dbc.Modal(
							[
								dbc.ModalHeader("Channel Specific Information"),
								dbc.ModalBody(
								id="window-info-modal-body"
								),
								dbc.ModalFooter(
									dbc.Button("Close", id="window-info-close", className="ml-auto")
								),
							],
							id="window-info-modal",
						),

					],
				),
				# div for everything in between two sidebars
				html.Div(
					className='nine columns trace-plot',
					children=[
						html.Div(
							className="container scalable",
							id="uploader-div",
							style={'display': 'none'}, 
							children=[
								# https://github.com/np-8/dash-uploader
								du.Upload(
									id='dash-uploader', 
									text='Drag and Drop Here to Upload!',
									# max_files=3, 
									default_style={
										"font-size": "2rem",
										"margin": "0px auto",
										"width": "30%",
										"min-height": "40px",
										"line-height": "40px",
										"text-align": "center",
										"border-width": "1px",
										"border-style": "dashed",
										"border-radius": "7px",
										"display" : "none"
									}),				
							],
						),
						dbc.Toast(
							id="upload-notify",
							header="Upload successful!",
							is_open=False,
							dismissable=True,
							icon="success",
							duration=4000,
							# top: 66 positions the toast below the navbar
							style={"position": "fixed", "top": 66, "right": 10, "width": 350, "font-size": "1.5rem"},
						),						
						html.Div(
							dcc.Graph(
								id="eeg-signal-graph",
								figure={},
							),
							className="container scalable eeg-signal-plot",
							id='eeg-signal-plot',
							style={'display': 'none'},
							# style={'display': 'none', 'autoscale': False, 'overflowY': 'scroll', 'height': 600},
						),
						html.Div(
							className="container scalable eeg-signal-plot",
							id='eeg-signal-plot-trace',
						),
						html.Div(
							className="container scalable eeg-signal-plot",
							id='eeg-signal-plot-preprocess',
						),						
					]
				),
				# div for right sidebar
				html.Div(
					className='two columns',
					id='node-interpret-graph-div',
					style={'display': 'none'},
					children=[
						html.Div(
							dcc.Graph(
								id='node-interpret-graph',
								figure={},
								# figure=display_psd_graph()
							),
							className="container scalable",
						)
					]
				),
				# div for everything below trace plot
				html.Div(
					className='twelve columns below-trace-plot container scalable',
					children=[
						html.Div(
							className="one column",
						),
						html.Div(
							className="nine columns offset-by-one",
							children=[
								html.Div(
									dcc.Graph(
										id='psd-graph',
										# NOTE: no input associated, direct call to function
										# figure=display_psd_graph()
										figure={}
									),
									id="psd-graph-div",
									className="container scalable",
									style={'display': 'none'},
									# className="five columns windows-table-div",
									# id = 'windows-table-div',

									# dash_table.DataTable(
									# 	id='windows-table',
									# 	columns=[{"name": i, "id": i} for i in ["Window Start", "Window End", "Prediction"]],
									# 	data= [],
									# 	row_selectable="single",
									# 	page_size= 5,
									# ),									
									# className="five columns windows-table-div",
									# id = 'windows-table-div',
									# style={'display': 'none'},
								),
								# html.Div(
								# 	className='four columns',
								# 	id='click-data', 
								# 	style={'color': 'white'},
								# 	children=[
								# 	dcc.Markdown("""
								# 		**Click Data**

								# 		Click on points in the graph.
								# 	""")],
								# ),
								# html.Div(
								# 	className="four columns node-graph",
								# 	id = 'node-graph',
								# 	style={'color': 'white'},
								# ),
							]
						),
						html.Div(
							className="two columns node-graph",
							id = 'node-graph',
							style={'color': 'white'},
						),
					]
				),
			]
		),
	]
)

# notify when file upload is complete
# https://github.com/np-8/dash-uploader/blob/master/docs/dash-uploader.md
@app.callback(
	[Output("upload-notify", "is_open"), Output("upload-notify", "children")],
    [Input('dash-uploader', 'isCompleted')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id')],
)
def notify(iscompleted, filenames, upload_id):
    #TODO: store the file path in server-side memory! idea - pickle by session ID
	if not iscompleted:
		return [False, ""]
	if filenames is not None:
		if upload_id:
			return [True, f"{str(Path(UPLOAD_FOLDER_ROOT))}\{upload_id}\{filenames[0]}"]
		else:
			return [True, f"{str(Path(UPLOAD_FOLDER_ROOT))}\{filenames[0]}"]
	return [False, ""]


# after upload is complete, show sensors available in file in the trace modal
@app.callback([Output('sensors-checklist', 'options'), 
				Output('sensors-checklist', 'value')],
			  [Input('trace-switch', 'on'), 
			#   Input('dash-uploader', 'isCompleted')
			  ],
			  [State('upload-notify', 'children'), 
			  State("sensors-checklist", "options"), State("sensors-checklist", "value")])
# def get_sensors_from_file(on, is_completed, raw_eeg_filepath, sensors_options, sensors_value):
def get_sensors_from_file(on, raw_eeg_filepath, sensors_options, sensors_value):
	# if on and is_completed and sensors_options == []:
	if on and sensors_options == []:
		options_list = [ ]
		value_list = [ ]

		# '/home/varatha2/projects/wagh/mne-dash-master/Dash_Uploads\\62d6a1fa-8dcd-11eb-9ad6-0242ac110002\\00010802_s001_t000.edf'
		# raw_eeg_filepath = raw_eeg_filepath.replace('\\', '/')

		raw_eeg_filepath = r"C:\Users\mzeng\Desktop\mne-dash-master\data\00010736_s001_t001.edf"


		# TODO: see if you can only read metadata without data
		raw_data = mne.io.read_raw_edf(raw_eeg_filepath, verbose=True).crop(tmax=1).resample(125.0)
		ch_names = raw_data.info['ch_names']

		for ch_name in ch_names:
			options_list.append({
				'label': ch_name,
				'value': ch_name
			})
		# value_list = ch_names
		return [options_list, value_list]
	else:
		return [sensors_options, sensors_value]


# TODO: accept multiple settings from sidebar as input for the trace plot output
# TODO: pickle the preprocessed file
# show the trace plot
@app.callback([
	Output('eeg-signal-graph', 'figure'), Output('eeg-signal-plot', 'style'),
	Output('psd-graph', 'figure'), Output('psd-graph-div', 'style'),
	],
# @app.callback(Output('eeg-signal-plot', 'children'),
			  [Input('trace-switch', 'on'), 
			#   Input('dash-uploader', 'isCompleted'), 
			  Input("trace-close", "n_clicks")],
			  [State('upload-notify', 'children'), State("trace-modal", "is_open"), State("sensors-checklist", "value"), State("preprocess-checklist", "value"), State('eeg-gcnn-switch', 'on')])
# def read_eeg_file(on, is_completed, n_clicks, raw_eeg_filepath, is_open, sensors_list, preprocessing_list, eeg_gcnn_on):
def read_eeg_file(on, n_clicks, raw_eeg_filepath, is_open, sensors_list, preprocessing_list, eeg_gcnn_on):
	# if on and n_clicks and is_completed:
	# if (on and is_completed) and (n_clicks and is_open):
	if (on) and (n_clicks and is_open):

		# '/home/varatha2/projects/wagh/mne-dash-master/Dash_Uploads\\62d6a1fa-8dcd-11eb-9ad6-0242ac110002\\00010802_s001_t000.edf'
		raw_eeg_filepath = raw_eeg_filepath.replace('\\', '/')

		# fig = go.Figure(layout=dict(xaxis=dict(title='time'), yaxis=dict(title='voltage')))
		# for ch_idx, ch_name in enumerate(raw_data.info['ch_names']):
		# 	fig.add_scatter(
		# 		x=times, 
		# 		y=data[ch_idx, :], 
		# 		name=ch_name,
		# 		mode='lines'
		# 	)

		SAMPLING_FREQ = 250.0
		WINDOW_LENGTH_SECONDS = 10.0
		WINDOW_LENGTH_SAMPLES = int(WINDOW_LENGTH_SECONDS * SAMPLING_FREQ)

		# CAUTION: for testing only
		raw_eeg_filepath = r"C:\Users\mzeng\Desktop\mne-dash-master\data\00010736_s001_t001.edf"

		# TODO: crop depending on file length!
		# FIXME: using entire file is too slow!!
		raw_data = mne.io.read_raw_edf(raw_eeg_filepath, verbose=True).crop(tmin=0, tmax=500).resample(SAMPLING_FREQ)
		# raw_data = mne.io.read_raw_edf(raw_eeg_filepath, verbose=True).resample(SAMPLING_FREQ)
		# n_channels = 15
		# data, times = raw_data.get_data(return_times=True)
		# data = data[:n_channels, :]
		# ch_names = raw_data.info['ch_names'][:n_channels]

		# step = 1. / n_channels
		# kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

		# # create objects for layout and traces
		# layout = Layout(yaxis=YAxis(kwargs), showlegend=True)
		# traces = [Scatter(x=times, y=data.T[:, 0])]

		# # loop over the channels
		# for ii in range(1, n_channels):
		# 		kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
		# 		layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
		# 		traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

		# # add channel names using Annotations
		# annotations = [Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
		# 									text=ch_name, font=Font(size=9), showarrow=False)
		# 						for ii, ch_name in enumerate(ch_names)]
		# layout.update(annotations=annotations)

		# # set the size of the figure and plot it
		# # layout.update(autosize=False, width=1000, height=600)
		# fig = Figure(data=traces, layout=layout)
		# fig.update_layout(
		# 	margin=dict(l=100, r=3, t=3, b=3),
		# 	# paper_bgcolor="LightSteelBlue",
		# )

		# # # Add shape regions
		# # fig.add_vrect(
		# # 	x0="20", x1="40",
		# # 	fillcolor="lightcoral", opacity=0.5,
		# # 	layer="below", line_width=0,
		# # ),

		# # fig.add_vrect(
		# # 	x0="60", x1="80",
		# # 	fillcolor="aquamarine", opacity=0.5,
		# # 	layer="below", line_width=0,
		# # )

		# # fig.add_vrect(
		# # 	x0="100", x1="120",
		# # 	fillcolor="lightcoral", opacity=0.5,
		# # 	layer="below", line_width=0,
		# # )

		# # return dcc.Graph(figure=fig, id="eeg-signal-graph")

		n_channels = len(sensors_list)
		step = 1. / n_channels

		# raw_data = mne.io.read_raw_edf(
		# 	"C:/tmp/Dash_Uploads/0fb10cba-5f64-11eb-8f25-1831bf439245/00003367_s003_t001.edf", 
		# 	verbose=True).crop(tmax=180).resample(250.0)

		raw_data_selected = raw_data.copy()
		raw_data_selected = raw_data_selected.pick_channels(sensors_list, ordered=True)
		data, times = raw_data_selected.get_data(return_times=True)
		ch_names = sensors_list
		# ch_names = raw_data.info['ch_names'][:n_channels]


		fig = make_subplots(
			rows=n_channels, cols=1,
			shared_xaxes=True,
			vertical_spacing=0.00,
		)

		for ch_idx, ch_name in enumerate(ch_names):
			Scatter(x=times, y=data.T[:, ch_idx])
			fig.add_trace(go.Scatter(x=times, y=data.T[:, ch_idx], mode='lines', name=ch_name, line=go.scatter.Line(color="black")), row=(ch_idx+1), col=1)

		# fig.add_trace(go.Scatter(x= [1, 1.75, 2.5, 3.5], y=[4, 2, 6, 3,  5], mode='lines'),
		#               row=2, col=1)
		# fig.add_trace(go.Scatter(x= [1, 1.5,  2, 2.5, 3, 3.5], y=[4, 2, 6, 3,  5, 0], mode='lines'),
		#               row=3, col=1)

		# add channel names using Annotations
		annotations = [Annotation(x=-0.07, y=0, xref='paper', yref='y%d' % (ii + 1),
											text=ch_name, font=Font(size=15), showarrow=False)
								for ii, ch_name in enumerate(ch_names)]

		# CAUTION: need a better way to handle annotation layers
		if eeg_gcnn_on:

			# compute all windows
			window_rows = [ ]
			for start_sample_index in range(0, int(int(raw_data.times[-1]) * SAMPLING_FREQ), WINDOW_LENGTH_SAMPLES):
				end_sample_index = start_sample_index + (WINDOW_LENGTH_SAMPLES - 1)
			
				# ensure 10 seconds are available in window and recording does not end
				if end_sample_index > raw_data.n_times:
					break

				# save metadata to row dict
				row = {}
				row["record_length_seconds"] = raw_data.times[-1]
				row["start_sample_index"] = start_sample_index
				row["end_sample_index"] = end_sample_index
				row["predicted_text_label"] = ""
				row["predicted_numeric_label"] = ""
				window_rows.append(row)

			window_df = pd.DataFrame(window_rows, columns=[
													"record_length_seconds", 
													"start_sample_index",
													"end_sample_index",
													"predicted_text_label",
													"predicted_numeric_label"])			
			print("window df generated!")

			# preprocess recording
			raw_data = standardize_sensors(raw_data, channel_config="01_tcp_ar", return_montage=True)
			# raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)
			raw_data = highpass(raw_data, 1.0)
			# raw_data = remove_line_noise(raw_data)
			print("recording preprocessed!")

			# compute all features
			feature_matrix = np.zeros((window_df.shape[0], 8*6))
			spec_coh_matrix = np.zeros((window_df.shape[0], 64))

			for window_idx in window_df.index.tolist():
				
				# get raw data for the window
				start_sample = window_df.loc[window_idx]['start_sample_index']
				stop_sample = window_df.loc[window_idx]['end_sample_index']
				window_data = raw_data.get_data(start=start_sample, stop=stop_sample)

				
				# CONNECTIVITY EDGE FEATURES - compute spectral coherence values between all sensors within the window
				from mne.connectivity import spectral_connectivity
				# required transformation for mne spectral connectivity API
				transf_window_data = np.expand_dims(window_data, axis=0)

				# the spectral connectivity of each channel with every other.
				for ch_idx in range(8):

					# https://mne.tools/stable/generated/mne.connectivity.spectral_connectivity.html#mne.connectivity.spectral_connectivity
					spec_conn, freqs, times, n_epochs, n_tapers = spectral_connectivity(data=transf_window_data, 
													method='coh', 
													indices=([ch_idx]*8, range(8)), 
													sfreq=SAMPLING_FREQ, 
					#                                   fmin=(1.0, 4.0, 7.5, 13.0, 16.0, 30.0), 
					#                                   fmax=(4.0, 7.5, 13.0, 16.0, 30.0, 40.0),
													fmin=1.0, fmax=40.0,
													faverage=True, verbose=False)

					#             print(np.squeeze(spec_conn))
					#             print(freqs)
					#             print(times)
					#             print(n_epochs)
					#             print(n_tapers)
					
					spec_coh_values = np.squeeze(spec_conn)
					assert spec_coh_values.shape[0] == 8
					
					# save to connectivity feature matrix at appropriate index
					start_edge_idx = ch_idx * 8
					end_edge_idx = start_edge_idx + 8
					spec_coh_matrix[window_idx, start_edge_idx:end_edge_idx] = spec_coh_values
				
				# PSD NODE FEATURES - derive total power in 6 brain rhythm bands for each montage channel
				from mne.time_frequency import psd_array_welch
				psd_welch, freqs = psd_array_welch(window_data, sfreq=SAMPLING_FREQ, fmax=50.0, n_per_seg=150, 
												average='mean', verbose=False)
				# Convert power to dB scale.
				psd_welch = 10 * np.log10(psd_welch)
				band_powers = get_brain_waves_power(psd_welch, freqs)
				assert band_powers.shape == (8, 6)

				# flatten all features, and save to feature matrix at appropriate index
				features = band_powers.flatten()
				feature_matrix[window_idx, :] = features
				print("window done!")

				
			# save precomputed features
			np.save("data/X_psd.npy", feature_matrix)
			np.save("data/X_spec_coh.npy", spec_coh_matrix)
			print("features pre-computed!")

			# get predictions for all windows
			REDUCED_SENSORS = True
			EXPERIMENT_NAME = "psd_gnn_shallow"
			FOLD_IDX = 0
			GPU_IDX = 0
			BATCH_SIZE = 512
			DEVICE = torch.device('cuda:{}'.format(GPU_IDX) if torch.cuda.is_available() else 'cpu')
			model = EEGGraphConvNet(reduced_sensors=REDUCED_SENSORS)
			checkpoint = torch.load("./model/{}_fold_{}.ckpt".format(EXPERIMENT_NAME, FOLD_IDX), map_location=DEVICE)
			model.load_state_dict(checkpoint['state_dict'])
			model = model.to(DEVICE).double()

			NUM_WORKERS = 6
			PIN_MEMORY = True
			heldout_test_dataset = EEGGraphDataset(X=feature_matrix, indices=window_df.index.tolist(), loader_type="unseen", 
											sfreq=SAMPLING_FREQ, transform=Compose([ToTensor()]))
			heldout_test_loader = DataLoader(dataset=heldout_test_dataset, batch_size=BATCH_SIZE, 
										shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

			y_probs = torch.empty(0, 2).to(DEVICE)
			y_pred = [ ]

			model.eval()
			with torch.no_grad():
				for i, batch in enumerate(heldout_test_loader):
					batch = batch.to(DEVICE, non_blocking=True)
					X_batch = batch
					outputs = model(X_batch.x, X_batch.edge_index, X_batch.edge_attr, X_batch.batch).float()

					_, predicted = torch.max(outputs.data, 1)
					y_pred += predicted.cpu().numpy().tolist()

					# concatenate along 0th dimension
					y_probs = torch.cat((y_probs, outputs.data), 0)

			# returning prob distribution over target classes, take softmax across the 1st dimension
			y_probs = torch.nn.functional.softmax(y_probs, dim=1).cpu().numpy()
			assert y_probs.shape[0] == window_df.shape[0]

			# add predictions to window_df
			assert len(y_pred) == window_df.shape[0]
			
			print("window predictions made!")
			
			# CAUTION: use for testing - y_pred = [1]*window_df.shape[0]
			print(y_probs)
			print(y_pred)

			window_df["predicted_numeric_label"] = y_pred
			window_df["predicted_text_label"] = ["diseased" if y==0 else "healthy" for y in y_pred]

			# NOTE: finally, plot all annotations
			for window_idx, row_dict in window_df.iterrows():

				start_second = int(row_dict["start_sample_index"] / SAMPLING_FREQ)
				end_second = start_second + 10
				window_prediction = row_dict["predicted_text_label"]
				annot_color = "aquamarine" if window_prediction == "healthy" else "lightcoral"

				fig.add_vrect(
					x0=str(start_second), x1=str(end_second),
					fillcolor=annot_color, opacity=0.3,
				#     annotation_text="normal", annotation_position="top left",
					layer="below", line_width=0,
				)

		# # FIXME: add channel shapes!
		# lst_shapes = [
		# 	go.layout.Shape(
		# 		type="rect",
		# 		xref='x', yref='y2',
		# 		x0=20, x1=30,
		# 		# y0=-0.1, y1=0.1,
		# 		fillcolor="LightSkyBlue", opacity=0.3,
		# 		layer="below",
		# 		line_width=0,
		# 	)
		# ]

		# fig.update_shapes(dict(xref='x', yref='y'))
		fig.update_layout(
						# width=2000, 
						# height=500,
						paper_bgcolor='rgba(255,255,255, 1.0)',
						plot_bgcolor='rgba(0,0,0,0.05)',
						# autosize=True,
						margin=dict(l=100, r=0, t=0, b=0),
						yaxis=dict(domain=[1 - step, 1]),
						annotations=annotations,
						# shapes=lst_shapes,
						showlegend=False)

		fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.1, rangeslider_range=[0, 10], 
						col=1, row=n_channels)
		fig.update_xaxes(showgrid=True, zeroline=False)
		fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

		# fig.update_layout(clickmode='event+select')
		fig.update_traces(marker_size=20)


		# create the psd z-score heatmap
		X_psd = np.load("./data/X_psd.npy", allow_pickle=True)
		n_windows = X_psd.shape[0]
		X_psd_reshaped = X_psd.reshape((n_windows, 8, 6))
		X_psd_reshaped_avg = np.mean(X_psd_reshaped, axis=1)

		X_psd_reshaped_avg.T[0, :] = stats.zscore(X_psd_reshaped_avg.T[0, :])
		X_psd_reshaped_avg.T[1, :] = stats.zscore(X_psd_reshaped_avg.T[1, :])
		X_psd_reshaped_avg.T[2, :] = stats.zscore(X_psd_reshaped_avg.T[2, :])
		X_psd_reshaped_avg.T[3, :] = stats.zscore(X_psd_reshaped_avg.T[3, :])
		X_psd_reshaped_avg.T[4, :] = stats.zscore(X_psd_reshaped_avg.T[4, :])
		X_psd_reshaped_avg.T[5, :] = stats.zscore(X_psd_reshaped_avg.T[5, :])

		psd_fig = px.imshow(X_psd_reshaped_avg.T, 
						labels=dict(x="Time (s)", y="Brain Rhythms", color="PSD z-score"),
						x=np.array(range(X_psd_reshaped_avg.T.shape[1])) * 10,
						y=[r"$\delta$", r'$\theta$', r'$\alpha$', r'lower $\beta$', r'higher $\beta$', r'$\gamma$'],
						# color_continuous_scale='RdBu_r', 
						color_continuous_scale=px.colors.diverging.RdBu,
						color_continuous_midpoint=0.0,
						aspect='equal')
		psd_fig.update_xaxes(side="top")

		return [
			fig, {'display': ''},
			psd_fig, {'display': ''},
		]
	
	# elif (on and is_completed) and (n_clicks and is_open) and (eeg_gcnn_on):
	# 	# TODO: check all annotation switches here, trace plot already exists.
	# 	# if eeg_gcnn_on:
	# 	figure.add_vrect(
	# 		x0="20", x1="30",
	# 		fillcolor="aquamarine", opacity=0.5,
	# 	#     annotation_text="normal", annotation_position="top left",
	# 		layer="below", line_width=0,
	# 	)

	# 	figure.add_vrect(
	# 		x0="100", x1="130",
	# 		fillcolor="lightcoral", opacity=0.5,
	# 		annotation_text="abnormal", annotation_position="top left",
	# 		layer="below", line_width=0,
	# 	)
	# 	return [figure, {'display': ''}]
	else:
		return [
			go.Figure(data=[go.Scatter(x=[], y=[])]), {'display': 'none'}, 
			go.Figure(data=[go.Scatter(x=[], y=[])]), {'display': 'none'}, 
		]


# remove file upload GUI after upload is complete
@du.callback(
    output=Output('uploader-div', 'style'),
    id='dash-uploader'
)
def hide_upload(filepaths):
	return {'display': 'none'}


# TODO: https://dash.plotly.com/cytoscape/events
# when eeg-gcnn switch is enabled, show the brain graph below trace plot
@app.callback(Output('node-graph', 'children'),
			  [Input('interpret-switch', 'on')],)
			#   [State('brain-graph', 'figure')])
def show_brain_connectivity(value):
	if value is True:
		nodes = [
			{
				'data': {'id': short, 'label': label},
				# 'position': {'x': 20*lat, 'y': -20*long},
				'position': {'x': 270*x, 'y': -270*y},
				'locked': True,
				# 'selected': False,
				# 'selectable': False,
				# 'grabbable': False,
				# 'classes': 'green triangle'
			}
			for short, label, x, y in (
				# https://github.com/sappelhoff/eeg_positions/blob/master/data/standard_1020_2D.tsv
				("C3", "C3",	-0.3249,	0.0000),
				("C4", "C4",	0.3249,	0.0000),
				("Cz", "Cz",	0.0000,	0.0000),
				("F3", "F3",	-0.2744,	0.3467),
				("F4", "F4",	0.2744,	0.3467),
				("F7", "F7",	-0.5879,	0.4270),
				("F8", "F8",	0.5879,	0.4270),
				("Fp1", "Fp1",	-0.2245,	0.6910),
				("Fp2", "Fp2",	0.2245,	0.6910),
				("Fpz", "Fpz",	0.0000,	0.7266),
				("Fz", "Fz",	0.0000,	0.3249),
				("O1", "O1",	-0.2245,	-0.6910),
				("O2", "O2",	0.2245,	-0.6910),
				("Oz", "Oz",	0.0000,	-0.7266),
				("P3", "P3",	-0.2744,	-0.3467),
				("P4", "P4",	0.2744,	-0.3467),
				("P7", "P7",	-0.5879,	-0.4270),
				("P8", "P8",	0.5879,	-0.4270),
				("Pz", "Pz",	0.0000,	-0.3249),
				("T7", "T7",	-0.7266,	0.0000),
				("T8", "T8",	0.7266,	0.0000)

				# ('la', 'Los Angeles', 34.03, -118.25),
				# ('nyc', 'New York', 40.71, -74),
				# ('to', 'Toronto', 43.65, -79.38),
				# ('mtl', 'Montreal', 45.50, -73.57),
				# ('van', 'Vancouver', 49.28, -123.12),
				# ('chi', 'Chicago', 41.88, -87.63),
				# ('bos', 'Boston', 42.36, -71.06),
				# ('hou', 'Houston', 29.76, -95.37)
			)
		]

		edges = [
			# {'data': {'source': source, 'target': target}}
			# for source, target in (
			# 	('van', 'la'),
			# 	('la', 'chi'),
			# 	('hou', 'chi'),
			# 	('to', 'mtl'),
			# 	('mtl', 'bos'),
			# 	('nyc', 'bos'),
			# 	('to', 'hou'),
			# 	('to', 'nyc'),
			# 	('la', 'nyc'),
			# 	('nyc', 'bos')
			# )
		]

		return [
			cyto.Cytoscape(
				id='cytoscape-node-graph',
				layout={'name': 'preset'},
				style={
				'width': '100%', 'height': '300px', 
				'background-color': '#E6E6E6', 'label': 'data(label)'},
				elements=edges+nodes,
				responsive=False,
				userZoomingEnabled=False,
				userPanningEnabled=False
			)			
		]
	elif value is False:
		return

# # when eeg-gcnn switch is enabled, predict the EEG-GCNN model on the entire signal, and display the windows table
# @app.callback(
# 	[Output('windows-table', 'data'), Output('windows-table-div', 'style')],
# 	[Input('eeg-gcnn-switch', 'on')],
# 	#   [State('brain-graph', 'figure')])
# )
# def populate_window_table(value):
# 	if value is True:
# 		# return html.Blockquote("Healthy")
# 		data= [ {'Window Start': 100, 'Window End': 200, 'Prediction': 'HEALTHY'}, 
# 							{'Window Start': 200, 'Window End': 300, 'Prediction': 'DISEASED'},
# 							{'Window Start': 300, 'Window End': 400, 'Prediction': 'DISEASED'}, 
# 							{'Window Start': 400, 'Window End': 500, 'Prediction': 'HEALTHY'}, 
# 							{'Window Start': 500, 'Window End': 600, 'Prediction': 'HEALTHY'}, 
# 							{'Window Start': 600, 'Window End': 700, 'Prediction': 'HEALTHY'}, 
# 							{'Window Start': 700, 'Window End': 800, 'Prediction': 'DISEASED'}, 
# 						]
# 		return [data, {'display': ''}]
# 	else:
# 		return [[], {'display': 'none'}]






# toggle the preprocessing modal window based on binary switch
@app.callback(
	Output("preprocessing-modal", "is_open"),
	[Input("preprocessing-switch", "on"), Input("preprocessing-close", "n_clicks")],
	[State("preprocessing-modal", "is_open")]
)
def toggle_preprocessing_modal(on, n_clicks, is_open):
	if on and is_open:
		return False
	if on:
		return True

# preprocess signal when options modal is closed
@app.callback(
	Output("eeg-signal-plot-preprocess", "children"),
	[Input("preprocessing-switch", "on"), Input("preprocessing-close", "n_clicks"), Input("preprocessing-modal", "is_open")],
	[State("preprocess-checklist", "value")]
)
def preprocess_signal(on, n_clicks, is_open, checklist_values):
	if on and n_clicks and not is_open:
		# TODO: 
		return [html.Div(checklist_values)]
	return is_open	


# toggle the sensor settings modal window based on binary switch
@app.callback(
	Output("trace-modal", "is_open"),
	[Input("trace-switch", "on"), Input("trace-close", "n_clicks")],
	[State("trace-modal", "is_open")]
)
def toggle_trace_modal(on, n_clicks, is_open):
	if on and is_open:
		return False
	if on:
		return True
	return is_open


# # choose sensors to plot when options modal is closed
# @app.callback(
# 	Output("eeg-signal-plot-trace", "children"),
# 	[Input("trace-switch", "on"), Input("trace-close", "n_clicks"), Input("trace-modal", "is_open")],
# 	[State("sensors-checklist", "value")]
# )
# def choose_sensors(on, n_clicks, is_open, checklist_values):
# 	if on and n_clicks and not is_open:
# 		return [html.Div(checklist_values)]
# 	return is_open	

# plot psd graph using data/X_psd.npy file
# @app.callback(
#     [Output('psd-graph', 'figure')],)
	# [Input('trace-switch', 'on'), Input('dash-uploader', 'isCompleted'), Input("trace-close", "n_clicks")],
	# [State('upload-notify', 'children'), State("trace-modal", "is_open"), State("sensors-checklist", "value"), State("preprocess-checklist", "value"), State('eeg-gcnn-switch', 'on')])
# def display_psd_graph():

# 	X_psd = np.load("./data/X_psd.npy", allow_pickle=True)
# 	n_windows = X_psd.shape[0]
# 	X_psd_reshaped = X_psd.reshape((n_windows, 8, 6))
# 	X_psd_reshaped_avg = np.mean(X_psd_reshaped, axis=1)

# 	X_psd_reshaped_avg.T[0, :] = stats.zscore(X_psd_reshaped_avg.T[0, :])
# 	X_psd_reshaped_avg.T[1, :] = stats.zscore(X_psd_reshaped_avg.T[1, :])
# 	X_psd_reshaped_avg.T[2, :] = stats.zscore(X_psd_reshaped_avg.T[2, :])
# 	X_psd_reshaped_avg.T[3, :] = stats.zscore(X_psd_reshaped_avg.T[3, :])
# 	X_psd_reshaped_avg.T[4, :] = stats.zscore(X_psd_reshaped_avg.T[4, :])
# 	X_psd_reshaped_avg.T[5, :] = stats.zscore(X_psd_reshaped_avg.T[5, :])

# 	fig = px.imshow(X_psd_reshaped_avg.T, color_continuous_scale='RdBu_r', aspect='equal')
# 	return fig
	# {
    #     'data': [{
    #         'type': 'heatmap',
    #         'z': [[row.get(c['id'], None) for c in columns] for row in rows],
    #         'x': [c['name'] for c in columns]
    #     }]
    # }


# when user clicks on any point on trace plot or a row of the table, display information about the corresponding 10-sec window/row in a the modal body
@app.callback(
    # Output('click-data', 'children'),
    Output('window-info-modal-body', 'children'),
    [Input('eeg-signal-graph', 'clickData'), 
	# Input('windows-table', 'selected_rows')
	],
)
# def display_click_data(clickData, selected_rows):
def display_click_data(clickData):
	# TODO:
	ctx = dash.callback_context
	if ctx.triggered:
		if ctx.triggered[0]['prop_id'] == "eeg-signal-graph.clickData":		
			return json.dumps(clickData, indent=2)
		elif ctx.triggered[0]['prop_id'] == "windows-table.selected_rows":
			return [html.Div(selected_rows[0])]


# when user clicks on any point on psd heatmap, display interpretability information about the corresponding 10-sec window/row in the right sidebar
@app.callback(
    # Output('click-data', 'children'),
    [
		Output('node-interpret-graph', 'figure'),
		Output('node-interpret-graph-div', 'style'),
	],		
    [
		Input('psd-graph', 'clickData'),
		Input('interpret-switch', 'on'),
	# Input('windows-table', 'selected_rows')
	],
)
# def display_click_data(clickData, selected_rows):
def display_click_data(clickData, interpret_on):
	# TODO:

	ctx = dash.callback_context

	if interpret_on is True and ctx.triggered:
		
		if ctx.triggered[0]['prop_id'] == "psd-graph.clickData":		
			
			# extract window index from click data
			# example - { "points": [ { "curveNumber": 1, "pointNumber": 23875, "pointIndex": 23875, "x": 95.5, "y": -2.1511166778549695e-05 } ] }
			print(json.dumps(clickData, indent=2))

			window_idx = int(clickData["points"][0]["x"] / 10.0)
			window_time_start = clickData["points"][0]["x"]
			window_time_end = window_time_start + 10
			print(window_idx)

			# TODO: compute IG attr values for the current window
			node_attributions = np.load("data/subject_node_attributions.npy", allow_pickle=True)

			# create plot/heatmap for clicked window
			data = node_attributions[window_idx, :].reshape(8, 6)
			fig = px.imshow(data,
							title=f"Signal Window: {window_time_start}s - {window_time_end}s",
							labels=dict(x="Brain Rhythms", y="Channels", color="Attribution"),
							x=[r"$\delta$", r'$\theta$', r'$\alpha$', r'lower $\beta$', r'higher $\beta$', r'$\gamma$'],
							y=[
											"F7-F3", "F8-F4",
											"T3-C3", "T4-C4",
											"T5-P3", "T6-P4",
											"O1-P3", "O2-P4"
											]
						)
			fig.update_xaxes(side="bottom")
			return [
				fig,
				{'display': ''}
			]

		else:
			return [
				{},
				{'display': 'none'}
			]
	else:
		return [
			{},
			{'display': 'none'}
		]



# toggle the sensor settings modal window based on binary switch
@app.callback(
	Output("window-info-modal", "is_open"),
	[Input('eeg-signal-graph', 'clickData'), 
	# Input('windows-table', 'selected_rows'), 
	Input("window-info-close", "n_clicks")],
	[State("window-info-modal", "is_open")]
)
# def toggle_window_info_modal(clickData, selected_rows, n_clicks, is_open):
def toggle_window_info_modal(clickData, n_clicks, is_open):
	ctx = dash.callback_context
	if ctx.triggered:
		if (clickData or selected_rows) and is_open:
			return False
		if clickData or selected_rows:
			return True
		return is_open


# TODO: data download - https://github.com/thedirtyfew/dash-extensions
# @app.callback(Output("download", "data"), [Input("btn", "n_clicks")])
# def download_text(n_clicks):
#     return dict(content="Hello world!", filename="hello.txt")


# Running the server
if __name__ == "__main__":
	app.run_server(debug=True) #, host="0.0.0.0", port=6010

# visdcc.Network(
# 		id = "eeg-gcnn-window-graph",
# 		options = {
# 			"autoResize": False,
# 			"height": '500px', 
# 			"width":'100%', 
# 			"physics": {'enabled': False}
# 		},
# 		data = {'nodes':[
# 						{'id': 1, 'label': 'Node 1', 'color':'#00ffff'},
# 						{'id': 2, 'label': 'Node 2'},
# 						{'id': 3, 'label': 'Node 3'},
# 						{'id': 4, 'label': 'Node 4'},
# 						{'id': 5, 'label': 'Node 5'},
# 						{'id': 6, 'label': 'Node 6'}
# 						],
# 				'edges':[
# 							{'id':'1-2', 'from': 1, 'to': 2},
# 							{'id':'1-3', 'from': 1, 'to': 3}, 
# 							{'id':'1-4', 'from': 1, 'to': 4}, 
# 							{'id':'1-5', 'from': 1, 'to': 5}, 
# 							{'id':'1-6', 'from': 1, 'to': 6},

# 							{'id':'2-3', 'from': 2, 'to': 3}, 
# 							{'id':'2-4', 'from': 2, 'to': 4}, 
# 							{'id':'2-5', 'from': 2, 'to': 5}, 
# 							{'id':'2-6', 'from': 2, 'to': 6},  
# 						]
# 				}
# 		),

# # get the vhdr file path
# def get_vhdr_path(full_filepaths):
# 	for path in full_filepaths:
# 		if '.vhdr' in path:
# 			return path
# 	return None

# DEFAULT_COLORSCALE = [[0, 'rgb(12,51,131)'], [0.25, 'rgb(10,136,186)'],
# 					  [0.5, 'rgb(242,211,56)'], [0.75, 'rgb(242,143,56)'], [1, 'rgb(217,30,30)']]

# DEFAULT_COLORSCALE_NO_INDEX = [ea[1] for ea in DEFAULT_COLORSCALE]

# drc = importlib.import_module("utils.dash_reusable_components")
# figs = importlib.import_module("utils.figures")

	# return html.Ul([html.Li(full_filepath)])
# @app.callback(Output('brain-graph', 'figure'),
# 			  [Input('slider-dataset-sample-size', 'value')],
# 			  [State('brain-graph', 'figure')])
# def update_graph(selected_dropdown_value, figure):
# 	index = (np.abs(stc.times * 1000 - selected_dropdown_value)).argmin()
# 	data = plotly_triangular_mesh(points, use_faces, smooth_mat * stc.data[:, index],
# 								  colorscale=DEFAULT_COLORSCALE, flatshading=False,
# 								  showscale=False, reversescale=False, plot_edges=False)
# 	figure["data"] = data
# 	figure["layout"] = plot_layout
# 	return figure


# @app.callback(Output('g1', 'figure'),
# 			  [Input('slider-dataset-sample-size', 'value')],
# 			  [State('g1', 'figure')])
# def update_graph(selected_dropdown_value, figure):
# 	index_time = (np.abs(stc.times * 1000 - selected_dropdown_value)).argmin()

# 	figure["data"][-1] = go.Scatter(
# 		x=[evoked.times[index_time] * 1000, evoked.times[index_time] * 1000],
# 		y=[-np.abs(evoked.data[2:306:3]).max(), np.abs(evoked.data[2:306:3]).max()],
# 		mode='lines',
# 		line=dict(color='white', width=6),
# 		hoverinfo='skip'
# 	)
# 	figure["layout"] = plot_layout_time
# 	return figure



		# html.Div(
		# 	id="body",
		# 	className="container scalable",
		# 	children=[
		# 		html.Div(
		# 			id="app-container",
		# 			# className="row",
		# 			children=[
		# 				html.Div(
		# 					# className="three columns",
		# 					id="left-column",
		# 					children=[
		# 						drc.Card(
		# 							id="first-card",
		# 							children=[
		# 								drc.NamedDropdown(
		# 									name="Select Subject",
		# 									id="dropdown-select-dataset",
		# 									options=[
		# 										{"label": "sample", "value": "sample"},
		# 									],
		# 									clearable=False,
		# 									searchable=False,
		# 									value="sample",
		# 								),
		# 								drc.NamedSlider(
		# 									name="Time",
		# 									id="slider-dataset-sample-size",
		# 									min=evoked.times.min() * 1000,
		# 									max=evoked.times.max() * 1000,
		# 									step=len(evoked.times),
		# 									marks={ii: '{0:.0f}'.format(ii) if ii == evoked.times[0] * 1000 else
		# 										'{0:.0f}'.format(ii) if not (i_l % 100) else ''
		# 										   for i_l, ii in enumerate(evoked.times * 1000)},
		# 									value=int(len(evoked.times) / 2),
		# 								),
		# 								drc.NamedSlider(
		# 									name="Threshold",
		# 									id="slider-dataset-noise-level",
		# 									min=0,
		# 									max=1,
		# 									marks={
		# 										i: str(i)
		# 										for i in [0, 0.25, 0.5, 0.75, 1]
		# 									},
		# 									step=0.1,
		# 									value=0.2,
		# 								),
		# 							],
		# 						),
		# 					],
		# 				), html.Div([
		# 					dcc.Graph(id='g1', figure={
		# 						'data': [go.Scatter(
		# 							x=evoked.times * 1000,
		# 							y=evoked.data[index, :].T,
		# 							mode='lines',
		# 							hoverinfo='x+y'
		# 						) for index in np.arange(2, 306, 3)] + [go.Scatter(
		# 							x=[evoked.times[index_time] * 1000, evoked.times[index_time] * 1000],
		# 							y=[-np.abs(evoked.data[2:306:3]).max(), np.abs(evoked.data[2:306:3]).max()],
		# 							mode='lines',
		# 							line=dict(color='white', width=6),
		# 							hoverinfo='skip'
		# 						)],
		# 						'layout': plot_layout_time, })
		# 				], className="six columns"),

		# 				html.Div(
		# 					[
		# 						dcc.Graph(
		# 							id="brain-graph",
		# 							figure={
		# 								"data": data,
		# 								"layout": plot_layout,
		# 							},
		# 							config={"editable": True, "scrollZoom": False},
		# 						)
		# 					],
		# 					className="graph__container",
		# 				),

		# 			],
		# 		)
		# 	],
		# ),
