"""Regression checks for generic live discovery and lossless local type mapping."""
from decimal import Decimal
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import bq_table_lists as lists
import bq_partition_to_local as partition
import bq_schema_to_local_pg as ddl
from bq_pg_types import postgres_type, value_for_field
from bq_table_lists import local_table_name
import sync_all_tables as sync
from copy_progress import CopyProgress


class PopulationTests(unittest.TestCase):
    def test_new_tables_and_views_are_discovered_on_each_run(self):
        client = Mock()
        client.list_tables.side_effect = [
            [SimpleNamespace(table_id='existing_mart')],
            [SimpleNamespace(table_id='new_dashboard_view'), SimpleNamespace(table_id='existing_mart')],
        ]
        with patch.dict(os.environ, {'BQ_PROJECT': 'test-project'}, clear=True):
            self.assertEqual(lists.load_layer_tables('mart_dev', client=client), ['existing_mart'])
            self.assertEqual(lists.load_layer_tables('mart_dev', client=client), ['existing_mart', 'new_dashboard_view'])
        self.assertEqual(client.list_tables.call_args.args, ('test-project.mart_dev',))

    def test_every_layer_uses_live_metadata(self):
        client = Mock()
        client.list_tables.return_value = []
        with patch.dict(os.environ, {'BQ_PROJECT': 'test-project', 'BQ_DATASET_GOLD': 'custom_marts'}, clear=True):
            for layer, dataset in [('landing', 'landing'), ('raw_dev', 'raw_dev'), ('staging_dev', 'staging_dev'), ('intermediate_dev', 'intermediate_dev'), ('mart_dev', 'custom_marts')]:
                lists.load_layer_tables(layer, client=client)
                self.assertEqual(client.list_tables.call_args.args, ('test-project.' + dataset,))

    def test_primary_dataset_environment_takes_precedence(self):
        with patch.dict(os.environ, {'BQ_DATASET_MART_DEV': 'primary', 'BQ_DATASET_GOLD': 'legacy'}, clear=True):
            self.assertEqual(lists.dataset_id_for_layer('mart_dev'), 'primary')

    def test_landing_entry_point_can_inject_a_client(self):
        client = Mock()
        client.get_table.return_value = SimpleNamespace(schema=[])
        config = {'bq_project': 'test-project', 'bq_dataset': 'landing',
                  'bq_table': 'empty', 'target_table': 'empty', 'target_schema': 'landing',
                  'local_db_url': 'postgresql://localhost/test', 'limit': 0, 'partition_filter': ''}
        self.assertEqual(partition.sync_table(config, client=client), 0)
        client.query.assert_not_called()

    def test_discovery_failure_is_not_hidden_by_stale_catalog(self):
        client = Mock()
        client.list_tables.side_effect = RuntimeError('no discovery access')
        with patch.dict(os.environ, {'BQ_PROJECT': 'test-project'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'no discovery access'):
                lists.load_layer_tables('mart_dev', client=client)

    def test_explicit_subset_is_the_only_discovery_override(self):
        client = Mock()
        with patch.dict(os.environ, {'BQ_MART_DEV_TABLES': ' second,first,second '}, clear=True):
            self.assertEqual(lists.load_layer_tables('mart_dev', client=client), ['first', 'second'])
        client.list_tables.assert_not_called()

    def test_uniform_row_limit_and_correct_default_dataset(self):
        calls = []
        def run(cmd, **kwargs):
            calls.append((cmd, kwargs['env']))
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, {'BQ_PARTITION_LIMIT': '25'}, clear=True), \
                patch.object(sys, 'argv', ['sync_all_tables.py', '--dataset', 'mart_dev', '--workers', '1']), \
                patch.object(sync, 'load_layer_tables', return_value=['new_mart', 'another_mart']), \
                patch.object(sync.subprocess, 'run', side_effect=run):
            sync.main()
        self.assertEqual(len(calls), 2)
        for cmd, env in calls:
            self.assertNotIn('--limit', cmd)
            self.assertEqual(env['BQ_PARTITION_LIMIT'], '25')
            self.assertEqual(env['BQ_DATASET'], 'mart_dev')

    def test_full_copy_is_default_without_an_arbitrary_outer_deadline(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(partition.get_config()['limit'], 0)
            with patch.object(sys, 'argv', ['sync_all_tables.py', '--dataset', 'mart_dev']), \
                    patch.object(sync, 'load_layer_tables', return_value=['new_mart']), \
                    patch.object(sync.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
                sync.main()
            self.assertIsNone(run.call_args.kwargs['timeout'])

    def test_streamed_refresh_copies_more_than_10000_rows_and_replaces_old_data(self):
        fields = [SimpleNamespace(name=name, field_type=kind, mode='NULLABLE')
                  for name, kind in [('id', 'INT64'), ('label', 'STRING'), ('currency', 'STRING')]]
        engine = create_engine('sqlite:///:memory:')
        with engine.begin() as conn:
            conn.execute(text('CREATE TABLE snapshot (id BIGINT, label TEXT, currency TEXT)'))
            conn.execute(text("INSERT INTO snapshot VALUES (-1, 'old', NULL)"))
            rows = ({'id': 9007199254740993 + i, 'label': str(i), 'currency': None} for i in range(10005))
            copied = partition.write_rows(rows, fields, conn, table='snapshot')
            self.assertEqual(copied, 10005)
            self.assertEqual(conn.execute(text('SELECT COUNT(*) FROM snapshot')).scalar_one(), 10005)
            self.assertEqual(conn.execute(text('SELECT MIN(id) FROM snapshot')).scalar_one(), 9007199254740993)
            self.assertEqual(conn.execute(text('SELECT MAX(id) FROM snapshot')).scalar_one(), 9007199254750997)
            self.assertEqual(conn.execute(text('SELECT COUNT(*) FROM snapshot WHERE currency IS NOT NULL')).scalar_one(), 0)
            self.assertEqual(str(inspect(conn).get_columns('snapshot')[2]['type']), 'TEXT')

    def test_heartbeat_reports_a_worker_waiting_for_network_without_new_rows(self):
        seen = threading.Event()
        messages = []
        def collect(message):
            messages.append(message)
            if '3/10' in message and 'reading BigQuery rows' in message:
                seen.set()
        with CopyProgress('mart_dev.large_table', interval=0.01, emit=collect) as progress:
            progress.update('reading BigQuery rows', copied=3, total=10)
            self.assertTrue(seen.wait(1), 'No heartbeat while worker was waiting')
            self.assertIn('rows staged', messages[-1])
            self.assertNotIn('committed', messages[-1])
            progress.update('committed', copied=10)
        self.assertIn('10/10 (100.0%)', messages[-1])
        self.assertIn('committed', messages[-1])

    def test_zero_limit_means_full_copy(self):
        config = {'bq_project': 'test-project', 'bq_dataset': 'mart_dev', 'bq_table': 'new_mart',
                  'partition_filter': '', 'limit': 0}
        self.assertNotIn('LIMIT', partition.build_query(config))
        config['limit'] = 25
        self.assertTrue(partition.build_query(config).endswith(' LIMIT 25'))
        config['limit'] = -1
        with self.assertRaises(ValueError): partition.build_query(config)

    def test_failed_import_does_not_stop_other_tables_or_report_success(self):
        imported = []
        def run(cmd, **kwargs):
            name = cmd[cmd.index('--table') + 1]
            imported.append(name)
            return SimpleNamespace(returncode=2 if name == 'bad_table' else 0)
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, 'argv', ['sync_all_tables.py', '--dataset', 'mart_dev', '--workers', '1']), \
                patch.object(sync, 'load_layer_tables', return_value=['bad_table', 'good_table']), \
                patch.object(sync.subprocess, 'run', side_effect=run):
            with self.assertRaises(SystemExit) as status:
                sync.main()
        self.assertEqual(status.exception.code, 1)
        self.assertEqual(imported, ['bad_table', 'good_table'])

    def test_schema_less_source_is_skipped_without_query_or_local_write(self):
        client = Mock()
        client.get_table.return_value = SimpleNamespace(schema=[])
        env = {'BQ_PROJECT': 'test-project', 'BQ_DATASET': 'raw_dev', 'LOCAL_DB_URL': 'postgresql://localhost/test'}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(sys, 'argv', ['bq_partition_to_local.py', '--table', 'empty_source']), \
                patch.object(partition, '_load_dotenv'), \
                patch('google.cloud.bigquery.Client', return_value=client), \
                patch.object(partition, 'get_engine') as engine:
            partition.main()
        client.query.assert_not_called()
        engine.assert_not_called()

    def test_null_only_columns_keep_source_types(self):
        self.assertEqual(ddl.bq_type_to_pg('STRING', 'NULLABLE'), 'TEXT')
        self.assertEqual(ddl.bq_type_to_pg('NUMERIC', 'NULLABLE'), 'NUMERIC')
        self.assertEqual(ddl.bq_type_to_pg('INT64', 'NULLABLE'), 'BIGINT')
        self.assertEqual(ddl.bq_type_to_pg('BOOL', 'NULLABLE'), 'BOOLEAN')
        self.assertEqual(ddl.bq_type_to_pg('STRING', 'REPEATED'), 'JSONB')

    def test_decimal_and_big_integer_values_retain_precision(self):
        for kind, value in [('NUMERIC', Decimal('1234567890123456789.123456789')), ('INT64', 9007199254740993)]:
            field = SimpleNamespace(field_type=kind, mode='NULLABLE')
            self.assertEqual(value_for_field(value, field), value)
            self.assertIsNone(value_for_field(None, field))

    def test_nested_json_is_not_double_encoded(self):
        field = SimpleNamespace(field_type='RECORD', mode='NULLABLE')
        self.assertEqual(value_for_field({'nested': [1, 2]}, field), {'nested': [1, 2]})
        field.field_type = 'JSON'
        self.assertEqual(value_for_field('{"count": 2}', field), {'count': 2})

    def test_long_names_do_not_collide_and_casing_is_preserved(self):
        prefix = 'same_prefix_' * 6
        a, b = local_table_name(prefix + 'first'), local_table_name(prefix + 'second')
        self.assertNotEqual(a, b)
        self.assertLessEqual(len(a.encode()), 63)
        self.assertLessEqual(len(local_table_name('é' * 70).encode()), 63)
        self.assertEqual(ddl.quote_ident('MixedCase'), '"MixedCase"')


if __name__ == '__main__':
    unittest.main()
