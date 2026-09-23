"""Agent-authored, reproducible pandas transformations, without Python eval/exec.

Only a small AST and typed dataframe/string methods are interpreted. No imports,
filesystem, network, Python introspection, callbacks, loops or arbitrary calls.
The trusted host alone loads the input and saves a new immutable dataset.
"""
import ast
import json
from pathlib import Path
import uuid

MAX_ROWS = 200000


def transform(frame, code):
    import pandas as pd
    if not isinstance(code, str) or not 0 < len(code) <= 16000:
        raise ValueError('预处理代码不能为空且不能超过 16000 字符')
    tree = ast.parse(code)
    if len(list(ast.walk(tree))) > 2000 or len(tree.body) > 80:
        raise ValueError('预处理代码超出单次操作限制')
    frame_methods = {'dropna', 'drop_duplicates', 'fillna', 'rename', 'reset_index', 'sort_values', 'copy', 'assign'}
    series_methods = {'fillna', 'astype', 'isin', 'notna', 'isna', 'clip', 'where', 'replace'}
    string_methods = {'strip', 'lstrip', 'rstrip', 'lower', 'upper', 'replace', 'slice', 'normalize', 'contains', 'len'}
    pandas_methods = {'to_datetime', 'to_numeric'}
    env = {'df': frame.copy()}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id == 'df':
            return env['df']
        if isinstance(node, ast.Subscript):
            obj, key = value(node.value), value(node.slice)
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj[key]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return ~value(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -value(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Add) and (isinstance(left, pd.Series) or isinstance(right, pd.Series)):
                return left + right
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left), value(node.comparators[0])
            if not isinstance(left, pd.Series):
                raise ValueError('筛选条件必须针对数据列')
            operators = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                         ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                         ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right}
            if type(node.ops[0]) in operators:
                return operators[type(node.ops[0])]()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == 'pd' and attr in pandas_methods:
                method = getattr(pd, attr)
            elif isinstance(owner, ast.Attribute) and owner.attr in {'str', 'dt'}:
                series = value(owner.value)
                allowed = string_methods if owner.attr == 'str' else {'strftime', 'floor'}
                if not isinstance(series, pd.Series) or attr not in allowed:
                    raise ValueError('不支持的列转换方法')
                method = getattr(getattr(series, owner.attr), attr)
            else:
                obj = value(owner)
                allowed = frame_methods if isinstance(obj, pd.DataFrame) else series_methods if isinstance(obj, pd.Series) else set()
                if attr not in allowed:
                    raise ValueError('预处理只允许登记的数据表与列方法')
                method = getattr(obj, attr)
            if any(k.arg is None or k.arg == 'inplace' for k in node.keywords):
                raise ValueError('请使用显式赋值，不支持展开参数或 inplace')
            return method(*[value(arg) for arg in node.args], **{k.arg: value(k.value) for k in node.keywords})
        raise ValueError('不支持的预处理语法；只允许 df 赋值、列筛选和登记的 pandas 转换，不执行任意 Python')

    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError('每一步请使用 df = … 或 df["列名"] = …')
        result, target = value(statement.value), statement.targets[0]
        if isinstance(target, ast.Name) and target.id == 'df' and isinstance(result, pd.DataFrame):
            env['df'] = result
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'df' and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
            env['df'][target.slice.value] = result
        else:
            raise ValueError('只能更新 df 或明确命名的数据列')
        if len(env['df']) > MAX_ROWS or len(env['df'].columns) > 500:
            raise ValueError('派生数据超出 20 万行 / 500 列限制')
    return env['df']


def preprocess(payload):
    import pandas as pd
    from ..capabilities import verify_dataset, dataset_import, dataset_profile
    from .readers import load_dataset
    source = payload['dataset']
    table = load_dataset(verify_dataset(source), profile_limit=MAX_ROWS)
    if table.rows_truncated:
        raise ValueError('预处理单次支持最多 20 万行；不会静默采样或截断，请明确选择样本')
    original = pd.DataFrame(table.rows, columns=table.columns)
    frame = transform(original, payload['code'])
    required = [payload['textColumn']] + ([payload['timeColumn']] if payload.get('timeColumn') else [])
    if frame.empty or any(column not in frame.columns for column in required):
        raise ValueError('预处理结果为空或缺少指定的正文 / 时间列')
    if frame.columns.duplicated().any():
        raise ValueError('派生数据不能有重复列名')
    text = frame[payload['textColumn']].fillna('').astype(str).str.strip()
    if not text.ne('').any():
        raise ValueError('预处理后没有有效正文，未切换训练数据')
    if payload.get('timeColumn'):
        dates = pd.to_datetime(frame[payload['timeColumn']], errors='coerce', format='mixed', utc=True)
        if dates.isna().any():
            raise ValueError('时间列仍有缺失或无法识别的值，请修正或明确筛选；不会编造时间')
    destination = Path(payload['home']) / 'preprocessing' / str(uuid.uuid4())
    destination.mkdir(parents=True)
    output = destination / '预处理数据.csv'
    frame.to_csv(output, index=False, encoding='utf-8-sig')
    # Full executable recipe for review/reproduction; the Agent snippet itself is
    # interpreted above and can never run these trusted I/O statements.
    script = destination / '预处理代码.py'
    script.write_text('import pandas as pd\nimport sys\nfrom pathlib import Path\nfrom workers.dataset.readers import load_dataset\n'
                      'table = load_dataset(Path(sys.argv[1]), profile_limit=200000)\n'
                      'if table.rows_truncated: raise ValueError("输入超过 20 万行，拒绝截断")\n'
                      'df = pd.DataFrame(table.rows, columns=table.columns)\n'
                      + payload['code'] + '\n'
                      + 'df.to_csv(sys.argv[2], index=False, encoding="utf-8-sig")\n', encoding='utf-8')
    dataset = dataset_import({'filePath': str(output), 'uploadDir': payload['uploadDir']})
    summary = {'sourceDatasetRef': source['datasetRef'], 'datasetRef': dataset['datasetRef'],
               'purpose': payload['purpose'], 'inputRows': len(original), 'outputRows': len(frame),
               'removedRows': len(original) - len(frame), 'columns': list(frame.columns),
               'textColumn': payload['textColumn'], 'timeColumn': payload.get('timeColumn'),
               'emptyTextRows': int(text.eq('').sum()), 'warnings': table.warnings}
    report = destination / '预处理校验.json'
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    profile = dataset_profile({'dataset': dataset})
    return {'dataset': dataset, 'summary': summary, 'profile': profile,
            'artifacts': [{'name': p.name, 'path': str(p)} for p in (script, report, output)]}
