import sys
sys.path.append('A_REDimplementation/A_RED')
sys.path.append('Spectrogram_A_RED')
from SpectrogramDataStream import SpectrogramDataStream, SpectrogramOracle
from A_RED import ARED
print('Imports successful')
ds = SpectrogramDataStream(max_samples=5, shuffle=False)
print('DataStream init OK, samples:', ds.n_samples)
print('First path sample:', ds.df.iloc[0]['spectrogram_npy_path'] if len(ds.df)>0 else 'N/A')
data = ds.stream_new_data_point()
print('Data point shape:', data.shape if hasattr(data,'shape') else len(data), 'dtype:', getattr(data,'dtype','N/A'))
print('Success - data loading works')
print('Oracle test:')
oracle = SpectrogramOracle(ds)
label, rel = oracle.answer_query(0)
print('Sample label from oracle:', label, 'relevance:', rel)
