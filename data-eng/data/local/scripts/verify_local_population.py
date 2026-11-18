#!/usr/bin/env python3
"""Compare a completed local refresh with live discovery, source types and import counts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from google.cloud import bigquery
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from bq_partition_to_local import _load_dotenv
from bq_pg_types import postgres_type
from bq_schema_to_local_pg import quote_ident
from bq_table_lists import LAYERS, dataset_id_for_layer, load_layer_tables, local_table_name
from engine_connector import get_engine

LOCAL = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=LAYERS, default='mart_dev')
    parser.add_argument('--log', type=Path, default=LOCAL / 'populate-last-run.log')
    args = parser.parse_args()
    _load_dotenv()
    project = os.environ['BQ_PROJECT']
    schema = dataset_id_for_layer(args.dataset)
    limit = int(os.environ.get('BQ_PARTITION_LIMIT', '0'))
    partition_filter = os.environ.get('BQ_PARTITION_FILTER', '').strip()
    client = bigquery.Client(project=project)
    selected = load_layer_tables(args.dataset, client=client)
    imported = {name: int(n) for n, name in re.findall(
        r'^Wrote (\d+) rows to .+ table (.+)$', args.log.read_text(), re.MULTILINE)}
    engine = get_engine()
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'project': project, 'dataset': schema, 'row_limit': limit, 'partition_filter': partition_filter,
              'discovered_objects': len(selected), 'objects': [], 'errors': []}
    with engine.connect() as conn:
        columns = {}
        for r in conn.execute(text('''SELECT c.relname, a.attname,
            format_type(a.atttypid, a.atttypmod)
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            JOIN pg_attribute a ON a.attrelid=c.oid
            WHERE n.nspname=:schema AND c.relkind IN ('r','p','v','m')
              AND a.attnum > 0 AND NOT a.attisdropped'''), {'schema': schema}):
            columns.setdefault(r[0], {})[r[1]] = r[2].lower()
        def metadata(name):
            return client.get_table(f'{project}.{schema}.{name}', timeout=15)
        with ThreadPoolExecutor(max_workers=4) as pool:
            tables = pool.map(metadata, selected)
            for table in tables:
                name = local_table_name(table.table_id)
                record = {'source': table.table_id, 'source_type': table.table_type,
                          'local_table': name, 'errors': []}
                report['objects'].append(record)
                if not table.schema:
                    record['status'] = 'skipped_no_schema'
                    continue
                expected = {f.name: str(postgres_type(f.field_type, f.mode).compile(
                    dialect=postgresql.dialect())).lower() for f in table.schema}
                actual = columns.get(name, {})
                if expected != actual:
                    record['errors'].append({'column_types': {
                        k: {'expected': expected.get(k), 'actual': actual.get(k)}
                        for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}})
                if name in columns:
                    n = conn.execute(text(f'SELECT COUNT(*) FROM {quote_ident(schema)}.{quote_ident(name)}')).scalar_one()
                    record['local_rows'] = n
                    logged = imported.get(f'{schema}.{name}')
                    record['rows_reported_by_import'] = logged
                    if n != logged:
                        record['errors'].append('Local row count differs from the completed import log')
                    if table.table_type == 'TABLE' and not partition_filter:
                        source_count = table.num_rows
                        record['source_rows_from_metadata'] = source_count
                        wanted = source_count if limit == 0 else min(source_count, limit)
                        if n != wanted:
                            record['errors'].append(f'Expected {wanted} rows from source metadata and configured cap')
                record['status'] = 'failed' if record['errors'] else 'passed'
                if record['errors']:
                    report['errors'].append({'source': table.table_id, 'errors': record['errors']})
    report['status'] = 'passed' if not report['errors'] else 'failed'
    report['verified_objects'] = sum(r['status'] == 'passed' for r in report['objects'])
    report['skipped_objects'] = sum(r['status'] == 'skipped_no_schema' for r in report['objects'])
    (LOCAL / 'populate-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Local population validation', '',
        f'Verified: {report["verified_at_utc"]}', f'Status: **{report["status"]}**', '',
        f'BigQuery source: `{project}.{schema}`.',
        f'Objects discovered: {len(selected)}; verified: {report["verified_objects"]}; skipped: {report["skipped_objects"]}.',
        f'Uniform row cap: {limit:,} per object (0 means all rows). No source-specific exceptions.', '',
        'Each run discovers current objects and replaces their local snapshots. New tables and views are picked up automatically. Views are materialized locally.',
        'Positive caps produce unordered samples. They do not guarantee newest observations, complete joins or agreement between fact samples and warehouse summary totals.', '',
        'Checks compare local table/column presence and PostgreSQL types with live BigQuery metadata, copied counts with the completed import log, and physical source-table counts with the row cap.',
        'BigQuery view counts use actual imported query results; metadata does not provide their logical row counts.', '',
        '| Source object | Type | Local rows | Result |', '| --- | --- | ---: | --- |']
    for r in report['objects']:
        lines.append(f'| `{r["source"]}` | {r["source_type"]} | {r.get("local_rows", "—")} | {r["status"]} |')
    lines += ['', '[Validation details](populate-validation.json)', '[Run log](populate-last-run.log)', '']
    (LOCAL / 'POPULATE_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in report.items() if k not in ('objects', 'errors')}, indent=2))
    if report['errors']: print(json.dumps(report['errors'], indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
