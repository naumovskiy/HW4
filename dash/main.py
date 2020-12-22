import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.graph_objs as go


def compare_columns(x):
    return sum(x[-3:]) / sum(x[:3])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

''' LOAD DATA '''
df_awr = pd.read_csv('data/awr_17122020.dat', header=0, sep='\t', parse_dates=['TT'])
df_fats = pd.read_csv('data/fat_sql_17122020.dat', header=0, sep='\t', parse_dates=['BEGIN_INTERVAL_TIME'])

''' WAIT CLASS & EVENTS '''
height_wc = 300
df_awr_simple = df_awr.groupby(['TT', 'WAIT_CLASS']).CNT.agg(['sum']).add_prefix('CNT_').reset_index()
colors_wait_class = {'Application': 'darkred', 'ONCPU': 'green',
                     'User I/O': 'blue', 'Commit': 'red', 'Other': 'magenta', 'Network': 'darkgreen',
                     'System I/O': 'darkblue', 'Configuration': 'chocolate', 'Concurrency': 'crimson'}
fig_wc = px.area(df_awr_simple, x="TT", y='CNT_sum', color='WAIT_CLASS', hover_name='WAIT_CLASS', log_y=True,
                 color_discrete_map=colors_wait_class, title='WAIT CLASS profile',
                 labels=dict(TT='date time from ASH history', CNT_sum='Count event by minute'), height=height_wc)
fig_wc.update_layout(margin=dict(l=10, r=10, t=50, b=0))

color_wc_sequence = ['darkred', 'red', 'chocolate', 'green', 'magenta', 'darkblue', 'blue', 'crimson', 'darkgreen']
fig_wc_pie = px.pie(df_awr_simple, names="WAIT_CLASS", values='CNT_sum', color='WAIT_CLASS', labels='WAIT_CLASS',
                    height=height_wc, width=300, color_discrete_sequence=color_wc_sequence,
                    title='WAIT CLASS distribution', hover_data=['WAIT_CLASS'])
fig_wc_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_wc_pie.update_layout(margin=dict(l=10, r=10, t=50, b=0))

df_awr_ev_simple = df_awr.groupby(['TT', 'WAIT_CLASS', 'EVENT_']).CNT.agg(['sum']).add_prefix('CNT_').reset_index()
df_awr_ev_simple['SNAP_ID'] = df_awr_ev_simple.TT.dt.hour + df_awr_ev_simple.TT.dt.minute % 15
fig_ev = px.area(df_awr_ev_simple, x="TT", y='CNT_sum', color='EVENT_', hover_name='WAIT_CLASS', log_y=True,
                 title='EVETNS profile', height=height_wc,
                 labels=dict(TT='date time from ASH history', CNT_sum='Count event by minute'))
fig_ev.update_layout(margin=dict(l=10, r=10, t=50, b=0))

cat = ['Application',	'Commit',	'Concurrency',	'Configuration',	'Network',	'ONCPU',	'Other',	'System I/O',	'User I/O']
fig_ev_table = px.area(df_awr_ev_simple[df_awr_ev_simple.WAIT_CLASS != 'ONCPU'], x="TT", y='CNT_sum', color='EVENT_',
                       facet_col="WAIT_CLASS", facet_col_wrap=2, category_orders={"WAIT_CLASS": cat}, height=600,
                       title='EVENTS distribution by WAIT_CLASS(without CPU)',
                       log_y=True, labels=dict(CNT_sum="", TT=""))
fig_ev_table.update_layout(margin=dict(l=0, r=0, t=50, b=0))

''' SQL OPERATION NAME & PLAN '''
height_sql_opname = 400
df_awr_sql_simple = df_awr.groupby(['TT', 'WAIT_CLASS', 'SQL_OPNAME']).CNT.agg(['sum']).add_prefix('CNT_').reset_index()
fig_sql_opname = px.bar(df_awr_sql_simple, x="TT", y='CNT_sum', color='SQL_OPNAME', title='SQL OPNAME profile',
                        labels=dict(TT='date time from ASH history', CNT_sum='Count SQL OPERATION by minute'),
                        height=height_sql_opname)
fig_sql_opname.update_layout(margin=dict(l=10, r=10, t=50, b=0))

fig_sql_opname_pie = px.pie(df_awr_sql_simple, names="SQL_OPNAME", values='CNT_sum',
                            color='SQL_OPNAME', hover_data=['SQL_OPNAME'], labels='SQL_OPNAME',
                            title='SQL OPNAME distribution', height=height_sql_opname, width=350)
fig_sql_opname_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_sql_opname_pie.update_layout(margin=dict(l=10, r=10, t=50, b=0))

df_awr['SQL_PLAN_OPTIONS'] = df_awr['SQL_PLAN_OPTIONS'].apply(lambda x: '' if x == 'EMPTY' else x)
df_awr['SQL_PLAN'] = df_awr['SQL_PLAN_OPERATION'] + ' ' + df_awr['SQL_PLAN_OPTIONS']
df_awr_sql_plan_simple = df_awr.groupby(['TT', 'WAIT_CLASS', 'SQL_PLAN', 'SQL_PLAN_OPERATION',
                                         'SQL_PLAN_OPTIONS']).CNT.agg(['sum']).add_prefix('CNT_').reset_index()
fig_sql_plan = px.area(df_awr_sql_plan_simple, x="TT", y='CNT_sum', color='SQL_PLAN', hover_name='SQL_PLAN_OPERATION',
                       title='SQL PLAN profile',
                       labels=dict(TT='date time from ASH history', CNT_sum='Count PLAN OPER by minute'),
                       height=height_sql_opname)
fig_sql_plan.update_layout(margin=dict(l=10, r=10, t=50, b=0))

''' FATs SQL_ID '''
threshold = 1.1
df_fats_group = df_fats.groupby(['SQL_ID', 'BEGIN_INTERVAL_TIME']).ELA.mean().unstack().add_prefix('TT_')
df_fats_group = df_fats_group.dropna(axis=0)
df_fats_group['diff3'] = df_fats_group.apply(lambda x: compare_columns(x), axis=1)
df_fats_simple = df_fats_group[df_fats_group.diff3 > threshold]
sql_id_fat = df_fats_simple.drop(columns=['diff3']).T.columns
df_temp = df_fats[df_fats.SQL_ID.isin(sql_id_fat)].\
    groupby(['BEGIN_INTERVAL_TIME', 'SQL_ID']).ELA.agg(['mean']).reset_index()

cols = 3
rows = int(np.ceil(float(len(sql_id_fat)/cols)))
fig_sql = make_subplots(rows=rows, cols=cols, subplot_titles=tuple(df_fats_simple.index))
for i, name_col in enumerate(sql_id_fat):
    fig_sql.add_trace(go.Scatter(x=list(df_temp[df_temp.SQL_ID == name_col]['BEGIN_INTERVAL_TIME']),
                                 y=list(df_temp[df_temp.SQL_ID == name_col]['mean']),
                                 name=name_col, mode='lines+markers'), int(i/rows+1), i % cols+1)
fig_sql.update_layout(legend_orientation="v", hovermode="x",
                      margin=dict(l=0, r=0, t=50, b=0), title='FATs SQL_IDs(ELA constant growth)', height=600)
fig_sql.update_traces(hoverinfo="all", hovertemplate="<br>ELA: %{y}")


app.layout = html.Div([
        html.Div(children=[
            html.Div([
                dcc.Graph(
                    id='graph_wc',
                    figure=fig_wc
                )
            ], className='five columns'),
            html.Div([
                dcc.Graph(
                    id='graph_wc_pie',
                    figure=fig_wc_pie
                )
            ], className='two columns'),
            html.Div([

                dcc.Graph(
                    id='graph_ev',
                    figure=fig_ev
                )
            ], className='five columns')
             ], className='row'),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='graph_ev_table',
                    figure=fig_ev_table
                )
            ], className='seven columns'),
            html.Div([
                dcc.Graph(
                    id='graph_sql_ids',
                    figure=fig_sql
                )
            ], className='five columns'),
        ], className='row'),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='graph_sql_opname',
                    figure=fig_sql_opname
                )
                ], className='four columns'),
            html.Div([
                dcc.Graph(
                    id='graph_sql_opname_pie',
                    figure=fig_sql_opname_pie
                )
            ], className='two columns'),
            html.Div([
                dcc.Graph(
                    id='graph_sql_plan',
                    figure=fig_sql_plan
                )
            ], className='six columns'),
            ], className='row')
], className='row')

if __name__ == '__main__':
    app.run_server(debug=True)
