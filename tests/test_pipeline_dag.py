"""
Tests for the Airflow DAG.

Airflow lives in its own virtual environment, so these tests skip when it is
absent rather than failing. When it is present they assert what the plan's
definition of done requires: the DAG imports cleanly, contains the expected
steps in the expected order, and keeps training logic out of the DAG file by
shelling out to project modules.
"""

import inspect
import unittest
from pathlib import Path

try:
    from airflow.models import DagBag

    AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    AIRFLOW_AVAILABLE = False


DAG_FOLDER = str(
    Path(__file__).resolve().parents[1]
    / "airflow"
    / "dags"
)

DAG_ID = "recommendation_pipeline"

EXPECTED_TASK_ORDER = [
    "validate_raw_data",
    "preprocess",
    "validate_processed_data",
    "train_sweep",
    "select_model",
    "build_faiss",
    "build_deployment_manifest",
    "publish_model",
]


@unittest.skipUnless(
    AIRFLOW_AVAILABLE,
    "Airflow is not installed in this environment.",
)
class TestPipelineDag(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Airflow 3 dropped DagBag's include_examples argument, and example
        # DAGs are off by default there. Pass it only when supported so the
        # test works against both major versions.
        parameters = inspect.signature(
            DagBag.__init__
        ).parameters

        options = {}

        if "include_examples" in parameters:
            options["include_examples"] = False

        cls.dag_bag = DagBag(
            dag_folder=DAG_FOLDER,
            **options,
        )

    def test_dag_imports_without_errors(self):
        self.assertEqual(
            self.dag_bag.import_errors,
            {},
        )

    def test_dag_is_registered(self):
        self.assertIn(
            DAG_ID,
            self.dag_bag.dags,
        )

    def test_contains_every_pipeline_step(self):
        dag = self.dag_bag.dags[DAG_ID]

        self.assertEqual(
            sorted(dag.task_ids),
            sorted(EXPECTED_TASK_ORDER),
        )

    def test_steps_run_in_order(self):
        dag = self.dag_bag.dags[DAG_ID]

        for upstream, downstream in zip(
            EXPECTED_TASK_ORDER,
            EXPECTED_TASK_ORDER[1:],
        ):
            self.assertEqual(
                [
                    task.task_id
                    for task in dag.get_task(
                        upstream
                    ).downstream_list
                ],
                [downstream],
                f"{upstream} must be followed by "
                f"{downstream}",
            )

    def test_first_step_has_no_upstream(self):
        dag = self.dag_bag.dags[DAG_ID]

        self.assertEqual(
            dag.get_task(
                EXPECTED_TASK_ORDER[0]
            ).upstream_list,
            [],
        )

    def test_tasks_shell_out_to_project_modules(self):
        # Keeping training logic out of the DAG is what lets any step be
        # reproduced by hand and keeps Airflow's environment independent of
        # the ML dependencies.
        dag = self.dag_bag.dags[DAG_ID]

        for task_id in EXPECTED_TASK_ORDER:
            command = dag.get_task(
                task_id
            ).bash_command

            self.assertIn(
                " -m src.",
                command,
                f"{task_id} should invoke a project module",
            )

    def test_tasks_run_from_the_repository_root(self):
        # Every script resolves data and model paths relative to the
        # repository root, so a wrong working directory would silently read
        # and write the wrong files.
        dag = self.dag_bag.dags[DAG_ID]

        expected_root = str(
            Path(__file__).resolve().parents[1]
        )

        for task_id in EXPECTED_TASK_ORDER:
            self.assertEqual(
                dag.get_task(task_id).cwd,
                expected_root,
            )

    def test_validation_steps_target_both_stages(self):
        dag = self.dag_bag.dags[DAG_ID]

        self.assertIn(
            "--stage raw",
            dag.get_task(
                "validate_raw_data"
            ).bash_command,
        )

        self.assertIn(
            "--stage processed",
            dag.get_task(
                "validate_processed_data"
            ).bash_command,
        )

    def test_manual_trigger_only(self):
        # The pipeline is triggered deliberately; a schedule would start
        # training runs during the build.
        dag = self.dag_bag.dags[DAG_ID]

        # Airflow 3 exposes `schedule`; Airflow 2 exposed
        # `schedule_interval`.
        schedule = getattr(
            dag,
            "schedule",
            getattr(
                dag,
                "schedule_interval",
                "<absent>",
            ),
        )

        self.assertIsNone(schedule)

        self.assertFalse(dag.catchup)


if __name__ == "__main__":
    unittest.main()
