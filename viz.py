import json
from datetime import datetime

import pandas as pd
import numpy as np
from imageio import imread
from redis import StrictRedis
from bokeh.plotting import figure, output_notebook, show, curdoc
from bokeh.models import ColumnDataSource


REDIS_HOST = '127.0.0.1'
REDIS_PORT = 9999

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

r = StrictRedis(host=REDIS_HOST, port=REDIS_PORT)

def load_ref_data():
    query = 'SELECT * FROM sniffing.ref_sniffer'
    ref = pd.read_gbq(query, project_id='presence-detection-204517')
    return ref

def load_beacon_pos():
    return json.load(open('out_pos.json'))

ref = load_ref_data()
pos = load_beacon_pos()

def get_last_states(noise=10):
    addrs = [t for t in r.scan_iter()]
    data = pd.DataFrame(list(map(lambda m: json.loads(m), r.mget(addrs))))
    #import ipdb; ipdb.set_trace()
    n_samples, _ = data.shape
    data['datetime'] = data['datetime'].apply(lambda dt: datetime.utcfromtimestamp(dt))
    data = data.merge(ref, how='left', on='sniffer_addr')
    data['num_beacon'] = data['mdns_name'].str.replace('beacon-|.local', '')
    data['x_beacon'] = data['num_beacon'].apply(lambda s: pos[s][0])
    data['y_beacon'] = data['num_beacon'].apply(lambda s: pos[s][1])
    data['x_beacon'] = data['x_beacon'] + np.random.randint(-noise, noise, size=n_samples)
    data['y_beacon'] = IMG_HEIGHT - data['y_beacon'] + np.random.randint(-noise, noise, size=n_samples)
    print(data[data['sniffer_addr'] == 'b8:27:eb:36:d8:2f'])
    return data

fig = figure(
    x_range=(0, IMG_WIDTH), 
    y_range=(0, IMG_HEIGHT),
    plot_width=int(IMG_WIDTH / 1.7),
    plot_height=int(IMG_HEIGHT / 1.7),
)

fig.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
fig.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
fig.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
fig.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
fig.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
fig.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels

fig.image_url(
    ['http://localhost:8000/out.png'],
    x=[0], y=[IMG_HEIGHT], w=[IMG_WIDTH], h=[IMG_HEIGHT]
)

source = ColumnDataSource(data=dict(x=[], y=[]))

fig.circle(
    x='x', y='y', color='red', line_color='yellow', size=20, alpha=0.5, source=source
)

def update():
    states = get_last_states(noise=20)
    data = dict(
        x=states['x_beacon'].values,
        y=states['y_beacon'].values
    )
    source.data = data

curdoc().add_root(fig)
curdoc().add_periodic_callback(update, 1000)
