import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MODELS_DIR))

from gpu_utils import append_gpu_argument, configure_cuda_visible_devices
from config import PipelineConfig, config_from_args, create_parser


def _main_tree():
    return ast.parse((MODELS_DIR / "main.py").read_text(encoding="utf-8"))


def _model_tree(filename):
    return ast.parse((MODELS_DIR / filename).read_text(encoding="utf-8"))


def _function(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(node):
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _named_calls(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def _keyword_names(call):
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _is_name_attribute(node, object_name, attribute_name):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == object_name
        and node.attr == attribute_name
    )


def _is_args_attribute(node, attribute_name):
    return _is_name_attribute(node, "args", attribute_name)


def _string_list_values(node):
    if not isinstance(node, ast.List):
        return []
    return [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _gpu_add_argument_call(tree):
    calls = [
        node
        for node in ast.walk(_function(tree, "parse_args"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--gpu"
    ]
    assert len(calls) == 1
    return calls[0]


def _assert_optional_gpu_argument(test_case, tree):
    call = _gpu_add_argument_call(tree)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    test_case.assertIsInstance(keywords["type"], ast.Name)
    test_case.assertEqual("int", keywords["type"].id)
    test_case.assertIsInstance(keywords["default"], ast.Constant)
    test_case.assertIsNone(keywords["default"].value)
    test_case.assertIsInstance(keywords["help"], ast.Constant)
    test_case.assertIn("Physical GPU", keywords["help"].value)
    test_case.assertIn("overrides CUDA_VISIBLE_DEVICES", keywords["help"].value)


def _cuda_environment_assignments(tree):
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and _is_name_attribute(target.value, "os", "environ")
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "CUDA_VISIBLE_DEVICES"
            ):
                assignments.append(node)
    return assignments


def _assert_prepare_gpu_propagation_structure(test_case, tree):
    main_function = _function(tree, "main")
    prepare_if = next(
        node
        for node in main_function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and _is_args_attribute(node.test, "prepare")
    )
    model_loop = next(
        node
        for node in prepare_if.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model_name"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "models"
    )
    command_if = next(
        node
        for node in model_loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "model_name"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "theta"
    )

    command_assignments = []
    for branch in (command_if.body, command_if.orelse):
        assignments = [
            node
            for node in branch
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cmd"
            and "prepare_data.py" in _string_list_values(node.value)
        ]
        test_case.assertEqual(1, len(assignments))
        test_case.assertNotIn("--gpu", _string_list_values(assignments[0].value))
        command_assignments.extend(assignments)

    append_statements = [
        node
        for node in model_loop.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "append_gpu_argument"
    ]
    subprocess_statements = [
        node
        for node in model_loop.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and _is_name_attribute(node.value.func, "subprocess", "run")
    ]

    test_case.assertEqual(1, len(append_statements))
    test_case.assertEqual(1, len(subprocess_statements))
    append_call = append_statements[0].value
    subprocess_call = subprocess_statements[0].value
    test_case.assertEqual(2, len(append_call.args))
    test_case.assertIsInstance(append_call.args[0], ast.Name)
    test_case.assertEqual("cmd", append_call.args[0].id)
    test_case.assertTrue(_is_args_attribute(append_call.args[1], "gpu"))
    test_case.assertGreater(
        model_loop.body.index(append_statements[0]),
        model_loop.body.index(command_if),
    )
    test_case.assertGreater(
        model_loop.body.index(subprocess_statements[0]),
        model_loop.body.index(append_statements[0]),
    )
    test_case.assertEqual(1, len(subprocess_call.args))
    test_case.assertIsInstance(subprocess_call.args[0], ast.Name)
    test_case.assertEqual("cmd", subprocess_call.args[0].id)

    for branch in (command_if.body, command_if.orelse):
        nested_calls = [
            node
            for statement in branch
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "append_gpu_argument"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and _is_name_attribute(node.func, "subprocess", "run")
                )
            )
        ]
        test_case.assertEqual([], nested_calls)

    return command_assignments


def _assert_theta_gpu_propagation_structure(test_case, tree):
    run_theta = _function(tree, "run_theta")
    command_assignments = [
        node
        for node in run_theta.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "cmd"
        and "main.py" in _string_list_values(node.value)
    ]
    test_case.assertEqual(1, len(command_assignments))
    command_assignment = command_assignments[0]
    test_case.assertNotIn("--gpu", _string_list_values(command_assignment.value))

    optional_flag_names = {
        "no_early_stopping",
        "skip_viz",
        "skip_eval",
    }
    optional_flag_statements = [
        node
        for node in run_theta.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and _is_args_attribute(node.test, node.test.attr)
        and node.test.attr in optional_flag_names
    ]
    test_case.assertEqual(
        optional_flag_names,
        {statement.test.attr for statement in optional_flag_statements},
    )
    for statement in optional_flag_statements:
        append_calls = [
            node
            for branch_statement in statement.body
            for node in ast.walk(branch_statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cmd"
            and node.func.attr == "append"
        ]
        test_case.assertEqual(1, len(append_calls))

    gpu_append_statements = [
        node
        for node in run_theta.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "append_gpu_argument"
    ]
    test_case.assertEqual(1, len(gpu_append_statements))
    gpu_append_statement = gpu_append_statements[0]
    gpu_append_call = gpu_append_statement.value
    test_case.assertEqual(2, len(gpu_append_call.args))
    test_case.assertIsInstance(gpu_append_call.args[0], ast.Name)
    test_case.assertEqual("cmd", gpu_append_call.args[0].id)
    test_case.assertTrue(_is_args_attribute(gpu_append_call.args[1], "gpu"))

    skip_train_statements = [
        node
        for node in run_theta.body
        if isinstance(node, ast.If)
        and _is_args_attribute(node.test, "skip_train")
    ]
    test_case.assertEqual(1, len(skip_train_statements))
    skip_train_statement = skip_train_statements[0]

    command_index = run_theta.body.index(command_assignment)
    gpu_append_index = run_theta.body.index(gpu_append_statement)
    skip_train_index = run_theta.body.index(skip_train_statement)
    test_case.assertLess(command_index, gpu_append_index)
    for statement in optional_flag_statements:
        test_case.assertGreater(run_theta.body.index(statement), command_index)
        test_case.assertLess(run_theta.body.index(statement), gpu_append_index)
    test_case.assertLess(gpu_append_index, skip_train_index)

    subprocess_calls_in_body = [
        node
        for statement in skip_train_statement.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_name_attribute(node.func, "subprocess", "run")
    ]
    subprocess_calls_in_orelse = [
        node
        for statement in skip_train_statement.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_name_attribute(node.func, "subprocess", "run")
    ]
    all_subprocess_calls = [
        node
        for node in ast.walk(run_theta)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_name_attribute(node.func, "subprocess", "run")
    ]
    test_case.assertEqual([], subprocess_calls_in_body)
    test_case.assertEqual(1, len(subprocess_calls_in_orelse))
    test_case.assertEqual(all_subprocess_calls, subprocess_calls_in_orelse)
    subprocess_call = subprocess_calls_in_orelse[0]
    test_case.assertEqual(1, len(subprocess_call.args))
    test_case.assertIsInstance(subprocess_call.args[0], ast.Name)
    test_case.assertEqual("cmd", subprocess_call.args[0].id)

    for branch in (skip_train_statement.body, skip_train_statement.orelse):
        nested_gpu_append_calls = [
            node
            for statement in branch
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "append_gpu_argument"
        ]
        test_case.assertEqual([], nested_gpu_append_calls)

    return command_assignment


class GPUEnvironmentTests(unittest.TestCase):
    def test_explicit_gpu_overrides_existing_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            visible = configure_cuda_visible_devices(1)

            self.assertEqual("1", visible)
            self.assertEqual("1", os.environ["CUDA_VISIBLE_DEVICES"])

    def test_omitted_gpu_preserves_existing_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            visible = configure_cuda_visible_devices(None)

            self.assertEqual("2,3", visible)
            self.assertEqual("2,3", os.environ["CUDA_VISIBLE_DEVICES"])

    def test_omitted_gpu_preserves_existing_empty_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            visible = configure_cuda_visible_devices(None)

            self.assertEqual("", visible)
            self.assertEqual("", os.environ["CUDA_VISIBLE_DEVICES"])

    def test_omitted_gpu_defaults_to_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            visible = configure_cuda_visible_devices(None)

            self.assertEqual("0", visible)
            self.assertEqual("0", os.environ["CUDA_VISIBLE_DEVICES"])

    def test_negative_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            configure_cuda_visible_devices(-1)


class GPUCommandTests(unittest.TestCase):
    def test_explicit_gpu_is_appended_to_child_command(self):
        command = ["python3", "main.py"]

        self.assertEqual(
            ["python3", "main.py", "--gpu", "2"],
            append_gpu_argument(command, 2),
        )

    def test_omitted_gpu_is_not_appended_to_child_command(self):
        command = ["python3", "main.py"]

        self.assertEqual(command, append_gpu_argument(command, None))


class MainGPUConfigurationTests(unittest.TestCase):
    def test_rank_zero_helper_broadcasts_serializable_status_and_failure(self):
        tree = _main_tree()
        helper = _function(tree, "run_rank_zero")
        call_attrs = {
            call.func.attr
            for call in ast.walk(helper)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        }
        self.assertIn("is_initialized", call_attrs)
        self.assertIn("broadcast_object_list", call_attrs)
        self.assertNotIn("barrier", call_attrs)
        self.assertTrue(any(isinstance(node, ast.Try) for node in ast.walk(helper)))
        self.assertTrue(
            any(
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "RuntimeError"
                for node in ast.walk(helper)
            )
        )
        payload_assignments = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "payload"
                for target in node.targets
            )
        ]
        self.assertGreaterEqual(len(payload_assignments), 1)
        initial_payload = payload_assignments[0].value
        self.assertIsInstance(initial_payload, ast.List)
        self.assertEqual(
            [True, None, ""],
            [element.value for element in initial_payload.elts],
        )
        self.assertIn("format_exc", {
            call.func.attr
            for call in ast.walk(helper)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        })

    def test_all_ranks_helper_gathers_local_failures_before_return(self):
        tree = _main_tree()
        helper = _function(tree, "run_all_ranks")
        call_attrs = {
            call.func.attr
            for call in ast.walk(helper)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        }
        self.assertIn("is_initialized", call_attrs)
        self.assertIn("get_world_size", call_attrs)
        self.assertIn("all_gather_object", call_attrs)
        self.assertNotIn("barrier", call_attrs)
        self.assertTrue(any(isinstance(node, ast.Try) for node in ast.walk(helper)))
        self.assertTrue(
            any(
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "RuntimeError"
                for node in ast.walk(helper)
            )
        )
        returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
        self.assertTrue(
            any(isinstance(node.value, ast.Name) and node.value.id == "local_result"
                for node in returns)
        )
        self.assertIn("format_exc", call_attrs)
        parameter_names = [argument.arg for argument in helper.args.args]
        self.assertIn("signature_getter", parameter_names)
        self.assertIn("expected_signature", parameter_names)
        self.assertTrue(
            any(
                isinstance(node, ast.Raise)
                and "signature" in ast.unparse(node).lower()
                for node in ast.walk(helper)
            )
        )

    def test_coordinated_helpers_replace_raw_barriers(self):
        tree = _main_tree()
        barrier_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_name_attribute(node.func, "dist", "barrier")
        ]
        self.assertEqual([], barrier_calls)

    def test_lora_save_and_load_use_coordinated_helpers(self):
        train_two_stage = _function(_main_tree(), "train_two_stage")
        rank_zero_calls = _named_calls(train_two_stage, "run_rank_zero")
        all_rank_calls = _named_calls(train_two_stage, "run_all_ranks")
        self.assertTrue(
            any(
                len(call.args) >= 4
                and isinstance(call.args[3], ast.Constant)
                and call.args[3].value == "save LoRA adapter"
                for call in rank_zero_calls
            )
        )
        load_helper_call = next(
            call for call in all_rank_calls
            if len(call.args) >= 3
            and isinstance(call.args[2], ast.Constant)
            and call.args[2].value == "load LoRA adapter"
        )
        load_assignment = next(
            node
            for node in ast.walk(train_two_stage)
            if isinstance(node, ast.Assign)
            and node.value is load_helper_call
        )
        target = load_assignment.targets[0]
        self.assertIsInstance(target, ast.Tuple)
        self.assertIsInstance(target.elts[0], ast.Name)
        self.assertEqual("embedding_model", target.elts[0].id)
        load_keywords = _keyword_names(load_helper_call)
        self.assertIn("expected_signature", load_keywords)
        self.assertIn("signature_getter", load_keywords)
        self.assertIn("adapter_signature", _call_names(train_two_stage))

    def test_bow_generation_and_loading_use_coordinated_helpers(self):
        run_train = _function(_main_tree(), "run_train")
        rank_zero_calls = _named_calls(run_train, "run_rank_zero")
        all_rank_calls = _named_calls(run_train, "run_all_ranks")
        self.assertTrue(
            any(
                len(call.args) >= 4
                and isinstance(call.args[3], ast.Constant)
                and call.args[3].value == "check BOW artifacts"
                for call in rank_zero_calls
            )
        )
        self.assertTrue(
            any(
                len(call.args) >= 4
                and isinstance(call.args[3], ast.Constant)
                and call.args[3].value == "generate BOW artifacts"
                for call in rank_zero_calls
            )
        )
        self.assertTrue(
            any(
                len(call.args) >= 3
                and isinstance(call.args[2], ast.Constant)
                and call.args[2].value == "load BOW artifacts"
                for call in all_rank_calls
            )
        )
        for call in all_rank_calls:
            keywords = _keyword_names(call)
            self.assertIn("signature_getter", keywords)
            self.assertIn("expected_signature", keywords)

    def test_save_results_timestamp_is_broadcast_from_rank_zero(self):
        run_train = _function(_main_tree(), "run_train")
        save_helper_call = next(
            call
            for call in _named_calls(run_train, "run_rank_zero")
            if len(call.args) >= 4
            and isinstance(call.args[3], ast.Constant)
            and call.args[3].value == "save training results"
        )
        timestamp_assignment = next(
            node
            for node in ast.walk(run_train)
            if isinstance(node, ast.Assign) and node.value is save_helper_call
        )
        self.assertTrue(
            any(isinstance(target, ast.Name) and target.id == "timestamp"
                for target in timestamp_assignment.targets)
        )

    def test_logging_is_idempotent_and_rank_zero_coordinated(self):
        setup_logging = _function(_main_tree(), "setup_logging")
        self.assertEqual(1, len(_named_calls(setup_logging, "run_rank_zero")))
        handler_loops = [
            node
            for node in ast.walk(setup_logging)
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Attribute) and child.attr == "handlers"
                for child in ast.walk(node.iter)
            )
        ]
        self.assertEqual(1, len(handler_loops))
        loop_calls = _call_names(handler_loops[0])
        self.assertIn("getattr", loop_calls)
        self.assertTrue(
            {"close", "removeHandler"} <= {
                call.func.attr
                for call in ast.walk(handler_loops[0])
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
            }
        )
        nonzero_log_path_messages = [
            node
            for node in ast.walk(setup_logging)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "info"
            and any(
                isinstance(arg, ast.JoinedStr)
                and "Log file:" in "".join(
                    value.value for value in arg.values
                    if isinstance(value, ast.Constant)
                )
                for arg in node.args
            )
        ]
        self.assertEqual(1, len(nonzero_log_path_messages))
        self.assertTrue(
            any(
                isinstance(parent, ast.If)
                and "is_main_process" in _call_names(parent.test)
                and nonzero_log_path_messages[0] in list(ast.walk(parent))
                for parent in ast.walk(setup_logging)
            )
        )

    def test_evaluation_and_visualization_use_rank_zero_helper(self):
        tree = _main_tree()
        run_pipeline = _function(tree, "run_pipeline")
        main_function = _function(tree, "main")
        operation_names = {
            call.args[3].value
            for function in (run_pipeline, main_function)
            for call in _named_calls(function, "run_rank_zero")
            if len(call.args) >= 4 and isinstance(call.args[3], ast.Constant)
        }
        self.assertTrue(
            {
                "evaluate training results",
                "visualize training results",
                "evaluate existing results",
                "visualize existing results",
            } <= operation_names
        )
        side_effect_functions = [
            node
            for function in (run_pipeline, main_function)
            for node in ast.walk(function)
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "evaluate_training_results",
                "visualize_training_results",
                "evaluate_existing_results",
                "visualize_existing_results",
            }
        ]
        self.assertEqual(4, len(side_effect_functions))
        for function in side_effect_functions:
            return_statement = function.body[-1]
            self.assertIsInstance(return_statement, ast.Return)
            self.assertIsInstance(return_statement.value, ast.Constant)
            self.assertIsNone(return_statement.value.value)

    def test_pipeline_propagates_ddp_context_to_training(self):
        tree = _main_tree()
        run_pipeline = _function(tree, "run_pipeline")
        parameter_names = [argument.arg for argument in run_pipeline.args.args]
        self.assertEqual(
            ["config", "logger", "skip_viz", "skip_eval", "local_rank", "world_size"],
            parameter_names,
        )
        train_call = _named_calls(run_pipeline, "run_train")
        self.assertEqual(1, len(train_call))
        self.assertEqual(
            ["config", "logger", "local_rank", "world_size"],
            [argument.id for argument in train_call[0].args],
        )
        main_call = _named_calls(_function(tree, "main"), "run_pipeline")
        self.assertEqual(1, len(main_call))
        keywords = _keyword_names(main_call[0])
        self.assertEqual("local_rank", keywords["local_rank"].id)
        self.assertEqual("world_size", keywords["world_size"].id)

    def test_sparse_bow_size_guard_and_canonical_representation(self):
        tree = _main_tree()
        run_train = _function(tree, "run_train")
        large_bow_if = next(
            node
            for node in ast.walk(run_train)
            if isinstance(node, ast.If)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "issparse"
                for call in ast.walk(node.test)
            )
            and any(
                isinstance(child, ast.Attribute) and child.attr == "nbytes"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        self.assertIn("issparse", {
            call.func.attr
            for call in ast.walk(large_bow_if.test)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        })
        self.assertTrue(any(isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                            for node in ast.walk(large_bow_if.test)))

        save_bow = _function(tree, "save_bow_artifacts")
        call_attrs = {
            call.func.attr
            for call in ast.walk(save_bow)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
        }
        self.assertIn("save_npz", call_attrs)
        self.assertIn("commit_bow_artifacts", _call_names(save_bow))
        source = (MODELS_DIR / "main.py").read_text(encoding="utf-8")
        self.assertNotIn(
            'np.save(os.path.join(config.bow_dir, "bow_matrix.npy")',
            source,
        )

        run_evaluation = _function(tree, "run_evaluation")
        self.assertIn("load_bow_artifacts", _call_names(run_evaluation))

        load_bow = _function(tree, "load_bow_artifacts")
        source = ast.unparse(load_bow)
        self.assertIn("BOW matrix shape mismatch", source)
        self.assertIn("vocab embedding shape mismatch", source)
        self.assertIn("vocab embedding dtype mismatch", source)
        self.assertIn("vocab count mismatch", source)
        self.assertIn(
            "canonical BOW matrix exists without a valid manifest",
            source,
        )
        self.assertIn("bow_matrix.npy", source)
        self.assertIn("theta.bow-artifacts.legacy-dense", source)

        bow_signature_check = next(
            node
            for node in ast.walk(run_train)
            if isinstance(node, ast.FunctionDef)
            and node.name == "bow_artifact_signature_if_valid"
        )
        self.assertIn("has_valid_bow_manifest", _call_names(bow_signature_check))
        self.assertIn("validate_bow_manifest", _call_names(bow_signature_check))

    def test_visualization_remains_best_effort_without_reraise(self):
        run_visualization = _function(_main_tree(), "run_visualization")
        handlers = [
            handler
            for node in ast.walk(run_visualization)
            if isinstance(node, ast.Try)
            for handler in node.handlers
        ]
        self.assertTrue(handlers)
        self.assertFalse(
            any(
                isinstance(node, ast.Raise)
                for handler in handlers
                for statement in handler.body
                for node in ast.walk(statement)
            )
        )

    def test_ddp_binds_local_rank_before_process_group_initialization(self):
        setup_ddp = _function(_main_tree(), "setup_ddp")
        calls = [
            node.value
            for node in setup_ddp.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(
            isinstance(calls[0].func, ast.Attribute)
            and isinstance(calls[0].func.value, ast.Attribute)
            and _is_name_attribute(calls[0].func.value, "torch", "cuda")
            and calls[0].func.attr == "set_device"
        )
        self.assertEqual(1, len(calls[0].args))
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual("local_rank", calls[0].args[0].id)
        self.assertTrue(
            isinstance(calls[1].func, ast.Attribute)
            and isinstance(calls[1].func.value, ast.Name)
            and calls[1].func.value.id == "dist"
            and calls[1].func.attr == "init_process_group"
        )

    def test_ddp_uses_global_rank_for_process_group(self):
        setup_ddp = _function(_main_tree(), "setup_ddp")
        global_rank_calls = [
            node
            for node in setup_ddp.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "get_global_rank"
        ]
        self.assertEqual([], global_rank_calls)

        init_call = next(
            node
            for node in ast.walk(setup_ddp)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dist"
            and node.func.attr == "init_process_group"
        )
        keywords = {keyword.arg: keyword.value for keyword in init_call.keywords}
        self.assertIsInstance(keywords["rank"], ast.Call)
        self.assertIsInstance(keywords["rank"].func, ast.Name)
        self.assertEqual("get_global_rank", keywords["rank"].func.id)
        self.assertEqual("local_rank", keywords["rank"].args[0].id)

    def test_ddp_sampler_and_main_process_checks_use_global_rank(self):
        tree = _main_tree()
        global_rank_functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_global_rank"
        ]
        self.assertEqual(1, len(global_rank_functions))
        get_global_rank = global_rank_functions[0]
        return_statement = next(
            node
            for node in get_global_rank.body
            if isinstance(node, ast.Return)
        )
        self.assertIsInstance(return_statement.value, ast.Call)
        self.assertEqual("int", return_statement.value.func.id)
        get_call = return_statement.value.args[0]
        self.assertTrue(_is_name_attribute(get_call.func.value, "os", "environ"))
        self.assertEqual("get", get_call.func.attr)
        self.assertEqual("RANK", get_call.args[0].value)
        self.assertEqual("local_rank", get_call.args[1].id)

        is_main_process = _function(tree, "is_main_process")
        self.assertIn("get_global_rank", _call_names(is_main_process))

        sampler_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DistributedSampler"
        ]
        self.assertEqual(3, len(sampler_calls))
        for call in sampler_calls:
            rank_value = next(
                keyword.value for keyword in call.keywords if keyword.arg == "rank"
            )
            self.assertIsInstance(rank_value, ast.Call)
            self.assertIsInstance(rank_value.func, ast.Name)
            self.assertEqual("get_global_rank", rank_value.func.id)
            self.assertEqual("local_rank", rank_value.args[0].id)

    def test_clean_command_has_no_gpu_attribute(self):
        args = create_parser().parse_args(
            ["clean", "--input", "in", "--output", "out"]
        )

        self.assertFalse(hasattr(args, "gpu"))

    def test_main_reads_exact_optional_gpu_argument(self):
        tree = _main_tree()
        configure_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_cuda_visible_devices"
        ]

        self.assertEqual(1, len(configure_calls))
        gpu_access = configure_calls[0].args[0]
        self.assertIsInstance(gpu_access, ast.Call)
        self.assertIsInstance(gpu_access.func, ast.Name)
        self.assertEqual("getattr", gpu_access.func.id)
        self.assertEqual(3, len(gpu_access.args))
        self.assertIsInstance(gpu_access.args[0], ast.Name)
        self.assertEqual("args", gpu_access.args[0].id)
        self.assertIsInstance(gpu_access.args[1], ast.Constant)
        self.assertEqual("gpu", gpu_access.args[1].value)
        self.assertIsInstance(gpu_access.args[2], ast.Constant)
        self.assertIsNone(gpu_access.args[2].value)

    def test_main_separates_ddp_and_single_gpu_setup(self):
        main_function = _function(_main_tree(), "main")
        distributed_if = next(
            node
            for node in main_function.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "is_distributed"
        )

        ddp_calls = set().union(
            *(_call_names(statement) for statement in distributed_if.body)
        )
        single_gpu_calls = set().union(
            *(_call_names(statement) for statement in distributed_if.orelse)
        )
        self.assertIn("setup_ddp", ddp_calls)
        self.assertNotIn("configure_cuda_visible_devices", ddp_calls)
        self.assertIn("configure_cuda_visible_devices", single_gpu_calls)
        self.assertNotIn("setup_ddp", single_gpu_calls)

    def test_gpu_setup_precedes_training_dispatch_and_cuda_check(self):
        tree = _main_tree()
        main_function = _function(tree, "main")
        distributed_index = next(
            index
            for index, node in enumerate(main_function.body)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "is_distributed"
        )
        dispatch_index = next(
            index
            for index, node in enumerate(main_function.body)
            if isinstance(node, ast.Try)
            and {"run_train", "run_pipeline"} <= _call_names(node)
        )
        run_train = _function(tree, "run_train")
        run_pipeline = _function(tree, "run_pipeline")
        setup_device_calls = [
            node
            for node in ast.walk(run_train)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setup_device"
        ]

        self.assertLess(distributed_index, dispatch_index)
        self.assertEqual(1, len(setup_device_calls))
        self.assertIn("run_train", _call_names(run_pipeline))

    def test_main_file_uses_only_crlf_line_endings(self):
        source = (MODELS_DIR / "main.py").read_bytes()
        bare_lf_count = source.count(b"\n") - source.count(b"\r\n")

        self.assertIn(b"\r\n", source)
        self.assertEqual(0, bare_lf_count)

    def test_pipeline_config_defaults_to_gpu_zero(self):
        self.assertEqual(0, PipelineConfig().gpu_id)

    def test_config_file_without_gpu_defaults_to_zero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            config_file.write("{}")
            config_file.flush()

            self.assertEqual(0, PipelineConfig.load(config_file.name).gpu_id)

    def test_parser_accepts_explicit_gpu(self):
        args = create_parser().parse_args(["pipeline", "--gpu", "2"])

        self.assertEqual(2, args.gpu)

    def test_gpu_argument_is_available_to_common_subcommands(self):
        for command in ("train", "evaluate", "visualize", "pipeline"):
            with self.subTest(command=command):
                args = create_parser().parse_args([command, "--gpu", "2"])

                self.assertEqual(2, args.gpu)

    def test_parser_defaults_gpu_to_none(self):
        args = create_parser().parse_args(["pipeline"])

        self.assertIsNone(args.gpu)

    def test_explicit_gpu_overrides_config(self):
        args = create_parser().parse_args(["pipeline", "--gpu", "2"])

        self.assertEqual(2, config_from_args(args).gpu_id)

    def test_omitted_gpu_preserves_config_file_value(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            config_file.write('{"gpu_id": 3}')
            config_file.flush()
            args = create_parser().parse_args(
                ["pipeline", "--config", config_file.name]
            )

            self.assertEqual(3, config_from_args(args).gpu_id)

    def test_negative_gpu_is_rejected_by_config(self):
        args = create_parser().parse_args(["pipeline", "--gpu", "-1"])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            config_from_args(args)


class ParentPipelineGPUPropagationTests(unittest.TestCase):
    def setUp(self):
        self.tree = _model_tree("run_pipeline.py")

    def test_pipeline_imports_shared_gpu_helpers(self):
        imported_names = set()
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "gpu_utils":
                imported_names.update(alias.name for alias in node.names)

        self.assertEqual(
            {"append_gpu_argument", "configure_cuda_visible_devices"},
            imported_names,
        )

    def test_pipeline_gpu_argument_is_optional_physical_device_override(self):
        _assert_optional_gpu_argument(self, self.tree)

    def test_pipeline_main_configures_environment_immediately_after_parsing(self):
        main_function = _function(self.tree, "main")
        configure_statement = main_function.body[1]

        self.assertIsInstance(configure_statement, ast.Expr)
        configure_call = configure_statement.value
        self.assertIsInstance(configure_call, ast.Call)
        self.assertIsInstance(configure_call.func, ast.Name)
        self.assertEqual("configure_cuda_visible_devices", configure_call.func.id)
        self.assertEqual(1, len(configure_call.args))
        self.assertTrue(_is_args_attribute(configure_call.args[0], "gpu"))
        self.assertEqual([], _cuda_environment_assignments(self.tree))

    def test_theta_child_command_uses_optional_gpu_helper(self):
        _assert_theta_gpu_propagation_structure(self, self.tree)

    def test_theta_gpu_structure_rejects_helper_nested_in_training_branch(self):
        insufficient_tree = ast.parse(
            """
def run_theta(args):
    cmd = ["main.py"]
    if args.no_early_stopping:
        cmd.append("--no_early_stopping")
    if args.skip_viz:
        cmd.append("--skip_viz")
    if args.skip_eval:
        cmd.append("--skip_eval")
    if args.skip_train:
        result = "skipped"
    else:
        append_gpu_argument(cmd, args.gpu)
        subprocess.run(cmd)
"""
        )

        with self.assertRaises(AssertionError):
            _assert_theta_gpu_propagation_structure(self, insufficient_tree)

    def test_both_prepare_commands_share_optional_gpu_append_before_execution(self):
        command_assignments = _assert_prepare_gpu_propagation_structure(
            self,
            self.tree,
        )

        self.assertEqual(2, len(command_assignments))

    def test_prepare_gpu_structure_rejects_helper_nested_in_theta_branch(self):
        insufficient_tree = ast.parse(
            """
def main():
    args = parse_args()
    if args.prepare:
        for model_name in models:
            if model_name == "theta":
                cmd = ["prepare_data.py"]
                append_gpu_argument(cmd, args.gpu)
            else:
                cmd = ["prepare_data.py"]
            subprocess.run(cmd)
"""
        )

        with self.assertRaises(AssertionError):
            _assert_prepare_gpu_propagation_structure(self, insufficient_tree)

    def test_theta_train_config_records_effective_visible_devices(self):
        run_theta = _function(self.tree, "run_theta")
        train_config = next(
            node.value
            for node in ast.walk(run_theta)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "train_config"
                for target in node.targets
            )
        )
        entries = {
            key.value: value
            for key, value in zip(train_config.keys, train_config.values)
            if isinstance(key, ast.Constant)
        }
        self.assertIn("cuda_visible_devices", entries)
        value = entries["cuda_visible_devices"]

        self.assertIsInstance(value, ast.Call)
        self.assertIsInstance(value.func, ast.Attribute)
        self.assertEqual("get", value.func.attr)
        self.assertTrue(_is_name_attribute(value.func.value, "os", "environ"))
        self.assertEqual(
            ["CUDA_VISIBLE_DEVICES", "0"],
            [argument.value for argument in value.args],
        )


class PrepareDataGPUPropagationTests(unittest.TestCase):
    def setUp(self):
        self.tree = _model_tree("prepare_data.py")

    def test_prepare_data_imports_environment_helper(self):
        imported_names = set()
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "gpu_utils":
                imported_names.update(alias.name for alias in node.names)

        self.assertIn("configure_cuda_visible_devices", imported_names)

    def test_prepare_data_gpu_argument_matches_parent_contract(self):
        _assert_optional_gpu_argument(self, self.tree)

    def test_prepare_data_main_uses_helper_immediately_after_parsing(self):
        main_function = _function(self.tree, "main")
        configure_statement = main_function.body[1]

        self.assertIsInstance(configure_statement, ast.Expr)
        configure_call = configure_statement.value
        self.assertIsInstance(configure_call, ast.Call)
        self.assertIsInstance(configure_call.func, ast.Name)
        self.assertEqual("configure_cuda_visible_devices", configure_call.func.id)
        self.assertEqual(1, len(configure_call.args))
        self.assertTrue(_is_args_attribute(configure_call.args[0], "gpu"))
        self.assertEqual([], _cuda_environment_assignments(self.tree))


class ShellGPUForwardingTests(unittest.TestCase):
    def _source(self, filename):
        return (SCRIPTS_DIR / filename).read_text(encoding="utf-8")

    def _section(self, source, start_marker, end_marker):
        return source.split(start_marker, 1)[1].split(end_marker, 1)[0]

    def test_training_wrappers_only_forward_explicit_gpu(self):
        for filename in (
            "train_theta.sh",
            "train_baseline.sh",
            "sweep_topics.sh",
        ):
            with self.subTest(filename=filename):
                source = self._source(filename)

                self.assertIn('GPU=""', source)
                self.assertNotRegex(source, r"(?m)^GPU=0$")
                self.assertRegex(
                    source,
                    r'if \[ -n "\$GPU" \]; then\s+.*--gpu',
                )

    def test_train_theta_forwards_optional_gpu_to_prepare_and_training(self):
        source = self._source("train_theta.sh")
        prepare_section = self._section(
            source,
            "# Auto-run data preparation if needed",
            "# Check vocab_embeddings",
        )
        training_section = self._section(
            source,
            "# Build command",
            "# Pass data_exp if using new exp structure",
        )

        self.assertIn('GPU_ARGS=()', source)
        self.assertIn('GPU_ARGS=(--gpu "$GPU")', source)
        self.assertIn('[[ "$GPU" =~ ^[0-9]+$ ]]', source)
        self.assertIn('"${GPU_ARGS[@]}"', prepare_section)
        self.assertRegex(
            training_section,
            r'if \[ -n "\$GPU" \]; then\s+CMD="\$CMD --gpu \$GPU"\s+fi',
        )

    def test_train_baseline_forwards_optional_gpu_to_prepare_and_training(self):
        source = self._source("train_baseline.sh")
        prepare_section = self._section(
            source,
            "# If no workspace, need to preprocess data",
            "# Find the newly created workspace",
        )
        training_section = self._section(
            source,
            "# Step 2: Build and execute training command",
            "# Add workspace_dir if provided",
        )

        self.assertIn('GPU_ENV_ARGS=()', source)
        self.assertIn('GPU_ENV_ARGS=("CUDA_VISIBLE_DEVICES=$GPU")', source)
        self.assertIn('[[ "$GPU" =~ ^[0-9]+$ ]]', source)
        self.assertIn('env "${GPU_ENV_ARGS[@]}" python -c', prepare_section)
        self.assertRegex(
            training_section,
            r'if \[ -n "\$GPU" \]; then\s+CMD="\$CMD --gpu \$GPU"\s+fi',
        )

    def test_sweep_forwards_same_optional_gpu_to_prepare_and_wrappers(self):
        source = self._source("sweep_topics.sh")
        self.assertIn('GPU_ARGS=()', source)
        self.assertIn('GPU_ARGS=(--gpu "$GPU")', source)

        prepare_section = self._section(
            source,
            'echo "  [Data] Running prepare_data.py for $DS ..."',
            'DATA_EXP=$(basename "$(_check_data_ready)")',
        )
        theta_section = self._section(
            source,
            'bash "$SCRIPT_DIR/train_theta.sh"',
            '&& SUCCESS_LIST+=("$DS/$MDL/K=$K")',
        )
        baseline_section = self._section(
            source,
            'bash "$SCRIPT_DIR/train_baseline.sh"',
            '&& SUCCESS_LIST+=("$DS/$MDL/K=$K")',
        )
        for section in (prepare_section, theta_section, baseline_section):
            self.assertIn('"${GPU_ARGS[@]}"', section)

    def test_quick_start_does_not_force_physical_gpu_zero(self):
        source = self._source("quick_start.sh")

        self.assertNotRegex(source, r"(?m)^GPU=")
        self.assertNotIn("--gpu)", source)
        self.assertNotIn("--gpu 0", source)
        self.assertNotRegex(source, r"\s--gpu(?:\s|$)")
