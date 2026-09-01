"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
from datetime import datetime
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    vid2tag = {}
    vid2type = {}
    vid2upload = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
            vid2tag[r['video_id']] = r.get('tag') or 'UNK'
            vid2type[r['video_id']] = r.get('video_type') or 'UNK'
            try:
                vid2upload[r['video_id']] = datetime.strptime(r.get('upload_dt', ''), '%Y-%m-%d').date()
            except ValueError:
                vid2upload[r['video_id']] = None

    user2activity = {}
    user_path = os.path.join(data_dir, 'user_features_pure.csv')
    if os.path.isfile(user_path):
        with open(user_path) as fh:
            for r in csv.DictReader(fh):
                user2activity[r['user_id']] = r.get('user_active_degree') or 'UNK'

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                event_date = datetime.strptime(r['date'], '%Y%m%d').date()
                upload_date = vid2upload.get(r['video_id'])
                upload_age_days = -1 if upload_date is None else max(0, (event_date - upload_date).days)
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             vid2tag.get(r['video_id'], 'UNK'), int(r['time_ms']),
                             max(0, min(23, int(r.get('hourmin') or 0) // 100)),
                             event_date.weekday(), upload_age_days,
                             vid2type.get(r['video_id'], 'UNK'),
                             user2activity.get(r['user_id'], 'UNK')))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
