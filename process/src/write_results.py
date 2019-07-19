# -*- coding: utf-8 -*-
"""
Get a csv file with cluster files
and write to db
"""

import os, sys
import psycopg2
import pandas as pd
from datetime import datetime as dt
# custom modules
project_dir = os.path.dirname(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from src.utils import get_abs_path, get_cmd_argv
from src import env


col_mapping = {
        'time_min': 'period_begin',
        'time_max': 'period_end',
        'cluster': 'id',
        'first_image': 'photo_begin',
        'last_image': 'photo_end',
        }

convert_timestamps = ['time_min', 'time_max']


def format_timestamp(seconds_since_epoch):
    '''
    From integer seconds since epoch convert to timestamp with time zone 
    format
    '''
    t = dt.fromtimestamp(seconds_since_epoch)
    return t.strftime('%Y-%m-%d %H:%M:%S')


class DBConnector:
    '''
    Connect and write to a database
    '''
    def __init__(self, connection_params):
        self.conn = psycopg2.connect(**db_params)
        self.cur = self.conn.cursor()
        self.query = 'INSERT INTO {table} ({cols}) VALUES {values};'
        
    
    def add_values(self, file, key_mapping, convert_timestamps):
        csv_cols = list(key_mapping.keys())
        self.tbl_cols = list(key_mapping.values())
        df = pd.read_csv(file)[csv_cols]
        assert len(df) > 0, f'Dataframe {file} is empty'
        for col in convert_timestamps:
            df[col] = df[col].apply(lambda x: format_timestamp(x))
        base_q = ','.join(['%s'] * len(self.tbl_cols))
        self.values_query = ','.join([self.cur.mogrify(f'({base_q})', 
                                          tuple(v)).decode('utf-8') 
                                            for v in df.values])
    
        
    def insert_values(self, table):
        query = self.query.format(table = table, cols=','.join(self.tbl_cols),
                                  values=self.values_query)
        try:
            self.cur.execute(query)
            self.conn.commit()
            print(f'Values written to {table}')
        except psycopg2.DatabaseError as e:
            print(e)
            self.conn.rollback()
        self.cur.close()
        self.conn.close()
        
if __name__ == '__main__':
    # get configs
    stage = get_cmd_argv(sys.argv, 1, 'test')
    configs = env.ENVIRON[stage]
    db_params = configs['DB_CONNECTION']
    clustering_results = configs['WRITE_SEEN_TIMES'].format(name=configs['NAME'])
    # write
    file = get_abs_path(__file__, clustering_results, depth=2)
    db = DBConnector(db_params)
    db.add_values(file, col_mapping, convert_timestamps)
    db.insert_values('visitors')
    