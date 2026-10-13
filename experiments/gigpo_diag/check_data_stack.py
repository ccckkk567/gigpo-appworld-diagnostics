"""Small local dependency check; these records are not environment rollouts."""
import json
import os
from pathlib import Path


def main():
    import datasets
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    versions = dict(numpy=np.__version__, pyarrow=pa.__version__, datasets=datasets.__version__, pandas=pd.__version__)
    expected = dict(numpy='1.26.4', pyarrow='19.0.1', datasets='3.6.0')
    for package, version in expected.items():
        if versions[package] != version:
            raise RuntimeError(f'Unexpected {package} version: {versions[package]}')
    # Exercise the same nested prompt/extra_info column shapes as prepare.py.
    data = datasets.Dataset.from_dict({'index': list(range(16))})
    data = data.map(lambda row: dict(data_source='text', prompt=[dict(role='user', content='')],
                                   ability='agent', extra_info=dict(split='dependency_check', index=row['index'])),
                    remove_columns=['index'])
    output = Path(os.environ['TMPDIR']) / 'data-stack-check.parquet'
    data.to_parquet(str(output))
    restored = datasets.Dataset.from_parquet(str(output))
    if restored.to_dict() != data.to_dict():
        raise RuntimeError('datasets Parquet round trip changed records')
    if len(pd.read_parquet(output)) != 16 or pq.read_table(output).num_rows != 16:
        raise RuntimeError('Pandas/PyArrow Parquet row count mismatch')
    np.testing.assert_array_equal(pa.array(np.arange(4)).to_numpy(), np.arange(4))
    result = dict(status='passed', versions=versions, rows=16, environment_rollouts=0, training_updates=0)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('DATA_STACK_CHECK_PASSED ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
