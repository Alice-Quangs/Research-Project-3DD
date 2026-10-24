"""Shared generated-row validation for both model training scripts."""
import json
import numpy as np
import pandas as pd
from config import CONFIG, FEATURE_SCHEMA_VERSION, get_feature_names
from bin_types import bin_types

def load_partition(path, partition, heuristic, manifest):
    names=list(get_feature_names(heuristic))
    meta=json.loads(path.with_suffix('.meta.json').read_text())
    if (meta.get('heuristic')!=heuristic or meta.get('partition')!=partition
            or meta.get('feature_names')!=names
            or meta.get('feature_schema_version')!=FEATURE_SCHEMA_VERSION
            or meta.get('bin_catalogue')!=bin_types):
        raise ValueError(f'{partition}: dataset metadata/schema/catalogue mismatch.')
    if meta.get('order_ids')!=manifest[partition]:
        raise ValueError(f'{partition}: dataset does not match saved split IDs/order.')
    settings=meta.get('config',{}).get('data',{})
    if settings.get('logging_policy')!='all_open_bins_before_placement' or settings.get('include_new_bin_rows') is not False:
        raise ValueError('Dataset must use agreed all-bin logging without new-bin rows.')
    columns=list(CONFIG.data.metadata_columns)+names+[CONFIG.data.target_column]
    frame=pd.read_csv(path,usecols=columns,dtype={'order_id':str})
    if len(frame)!=meta.get('rows') or frame.empty:
        raise ValueError(f'{partition}: row count mismatch or empty dataset.')
    if frame['order_id'].isna().any() or not set(frame['order_id']).issubset(set(manifest[partition])):
        raise ValueError(f'{partition}: unexpected order IDs.')
    if frame.duplicated(list(CONFIG.data.metadata_columns)).any():
        raise ValueError(f'{partition}: duplicate candidate rows.')
    X=frame[names].to_numpy(dtype=np.float64)
    y=frame[CONFIG.data.target_column].to_numpy()
    if not np.isfinite(X).all() or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError(f'{partition}: finite features and both binary classes are required.')
    y=y.astype(int)
    if int(y.sum())!=meta.get('positive_rows') or int((y==0).sum())!=meta.get('negative_rows'):
        raise ValueError('Class counts differ from generated metadata.')
    return X,y,frame[list(CONFIG.data.metadata_columns)],meta

