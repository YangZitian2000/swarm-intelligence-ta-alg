import pandas as pd

file_path = 'test/scene_data.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df = df.set_index('id')

def get_agent_code(scene_code):
    return df.loc[scene_code]['agent']

def get_task_code(scene_code):
    return df.loc[scene_code]['task']
