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
if project_dir not in sys.path: sys.path.insert(0, project_dir)
from src.utils import get_abs_path, get_cmd_argv
from src.env import configs


FILE_DEPTH = 2

# create a mapping from csv file colnames to db columns
# as {csv col: db table col}
visitors_mapping = {
        'time_min': 'period_begin',
        'time_max': 'period_end',
        'cluster': 'id',
        'photo_begin': 'photo_begin',
        'photo_end': 'photo_end',
        }
users_mapping = {
        'age': 'age',
        'gender': 'gender',
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
        self.query = {
                'visitors': 'INSERT INTO visitors ({cols}) VALUES {values}',
                'users': 'INSERT INTO users ({cols}) VALUES {values} '
                            'RETURNING id'
                    }
        
    
    def add_values(self, csv_file, key_mapping, 
                   convert_timestamps=None, extend_table={}):
        '''
        Add values for insertion into the DB
        '''
        assert os.path.exists(csv_file), f'{csv_file} does not exist'
        # load and transform values for visitors table
        csv_cols = list(key_mapping.keys())
        self.db_cols = list(key_mapping.values())
        df = pd.read_csv(csv_file, usecols=csv_cols)
        assert len(df) > 0, f'Dataframe {csv_file} is empty'
        if convert_timestamps:
            for col in convert_timestamps:
                df[col] = df[col].apply(lambda x: format_timestamp(x))
        # add more cols if provided
        for col,vals in extend_table.items():
            assert len(vals) == len(df), 'Extra values should be equal to df len'
            df[col] = vals
            self.db_cols.append(col)
            csv_cols.append(col)
        # create DB queries
        base_q = ','.join(['%s'] * len(self.db_cols))
        self.insert_query = ','.join([self.cur.mogrify(f'({base_q})', 
                                      tuple(v)).decode('utf-8') 
                                        for v in df[csv_cols].values])
    
        
    def insert_values(self, table_name, return_result=False):
        '''
        Insert values to the database.
        Use after self.add_values()
        '''
        db_tbls = list(self.query.keys())
        assert table_name in db_tbls, f'table_name has to be among {db_tbls}'
        # insert clusters into visitors
        exec_query = self.query[table_name].format(cols=','.join(self.db_cols),
                                                   values=self.insert_query)
        try:    
            self.cur.execute(exec_query)
            if return_result:
                res = self.cur.fetchall()
            self.conn.commit()
            print(f'Values written to {table_name}')
        except psycopg2.DatabaseError as e:
            print(e)
            self.conn.rollback()
        if return_result:
            return res
        else:
            return None
            
            
    def close_connection(self):
        self.cur.close()
        self.conn.close()
        
'''
# load and transform values for users table
        if demographics_file:
            demogr_cols = list(demographics_key_mapping.keys()) + ['cluster']
            self.users_cols = list(demographics_key_mapping.values())
            users = pd.read_csv(demographics_file, usecols=demogr_cols)
            assert len(users) == len(visitors), \
                f'Demographics has {len(users)} rows whereas main df {len(visitors)}'
            matches = [a == b for a,b in zip(visitors['cluster'], users['cluster'])]
            assert sum(matches) == len(users), \
                    f'Cluster and Demographics lists are not ordered identically'
            del users['cluster']
'''
        
if __name__ == '__main__':
    # get configs
    q_name = get_cmd_argv(sys.argv, 2, 'test')
    q_date = get_cmd_argv(sys.argv, 1, None)
    db_params = configs['DB_CONNECTION']
    clustering_results = get_abs_path(__file__, 
                                      configs['WRITE_RESULTS'].format(name=q_name,
                                                         date=q_date),
                                     depth=FILE_DEPTH)
    demographics = get_abs_path(__file__, 
                                configs['WRITE_DEMOGRAPHICS'].format(name=q_name,
                                                             date=q_date),
                                depth=FILE_DEPTH)
    # db worker
    db = DBConnector(db_params)
    # add demographics to USERS
    db.add_values(demographics, users_mapping)
    user_ids = db.insert_values('users', return_result=True)
    extend_vals = {'user_id': [u[0] for u in user_ids]}
    # add clusters to VISITORS
    db.add_values(clustering_results, visitors_mapping, 
                  convert_timestamps=convert_timestamps, 
                  extend_table=extend_vals)
    db.insert_values('visitors')
    db.close_connection()
    