import sys
import json
import logging
from os import environ
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from redis import StrictRedis
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource


REDIS_HOST = environ['REDIS_HOST']
REDIS_PORT = int(environ['REDIS_PORT'])

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(module)s|%(name)s|%(filename)s - %(levelname)s @ %(asctime)s : %(message)s'
)

r = StrictRedis(host=REDIS_HOST, port=REDIS_PORT)

def load_ref_data():
    query = 'SELECT * FROM sniffing.ref_sniffer'
    ref = pd.read_gbq(query, project_id='presence-detection-204517')
    return ref

def load_beacon_pos():
    return json.load(open('beacons.json'))

ref = load_ref_data()
pos = load_beacon_pos()

def get_last_states(noise=10, max_delta=timedelta(minutes=10)):
    addrs = [t for t in r.scan_iter()]
    if not addrs:
        return None
    data = pd.DataFrame(list(map(lambda m: json.loads(m), r.mget(addrs))))
    data['datetime'] = data['datetime'].apply(lambda dt: datetime.utcfromtimestamp(dt))
    data = data[data['datetime'] >= datetime.utcnow() - max_delta]
    n_samples, _ = data.shape
    data = data.merge(ref, how='left', on='sniffer_addr')
    data['num_beacon'] = data['mdns_name'].str.replace('beacon-|.local', '')
    data['x_beacon'] = data['num_beacon'].apply(lambda s: pos[s][0])
    data['y_beacon'] = data['num_beacon'].apply(lambda s: pos[s][1])
    data['x_beacon'] = data['x_beacon'] + np.random.randint(-noise, noise, size=n_samples)
    data['y_beacon'] = IMG_HEIGHT - data['y_beacon'] + np.random.randint(-noise, noise, size=n_samples)
    return data

def compare(prev, cur):
    prev = prev[['adv_addr', 'sniffer_addr']]
    cur = cur[['adv_addr', 'sniffer_addr']]
    data = pd.merge(prev, cur, on='adv_addr', suffixes=('_prev', '_cur'))
    n_stay = (data['sniffer_addr_prev'] == data['sniffer_addr_cur']).sum()
    n_moved = (data['sniffer_addr_prev'] != data['sniffer_addr_cur']).sum()
    if n_moved > 0:
        logging.info(f'{n_stay} stayed, {n_moved} moved (total = {n_moved + n_stay})')
        sys.stdout.flush()

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
    x='x', y='y', color='red', line_color='yellow', size=10, alpha=0.5, source=source
)

prev_states = None

def update():
    global prev_states
    states = get_last_states(noise=20)
    if states is None:
        return
    logging.info(f'min time: {states["datetime"].min()}')
    logging.info(f'max time: {states["datetime"].max()}')
    data = dict(
        x=states['x_beacon'].values,
        y=states['y_beacon'].values
    )
    source.data = data

    if prev_states is not None:
        compare(prev_states, states)
    prev_states = states

curdoc().add_root(fig)
curdoc().add_periodic_callback(update, 1000)
