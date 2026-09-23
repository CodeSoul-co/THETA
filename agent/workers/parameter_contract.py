"""Read CLI enum contracts without importing or running the numerical engine."""
import ast


def parameter_choices(root, flags):
    choices = {}
    for filename in ('prepare_data.py', 'run_pipeline.py'):
        tree = ast.parse((root / 'src/models' / filename).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
                continue
            names = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            enum = next((item.value for item in node.keywords if item.arg == 'choices'), None)
            if enum is None:
                continue
            try:
                values = {item for item in ast.literal_eval(enum) if item is not None}
            except (ValueError, TypeError):
                # Dynamic constants are not evaluated; another entry may declare the literal contract.
                continue
            for key, flag in flags.items():
                if flag in names:
                    choices[key] = choices[key] & values if key in choices else values
    return {key: sorted(values) for key, values in choices.items()}
